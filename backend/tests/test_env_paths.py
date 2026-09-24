"""HF_HOME from .env is resolved against the repository root, not the cwd.

.env ships HF_HOME=./data/hf-cache, where setup caches Twitter/twhin-bert-base.
The backend runs in backend/ and each simulation in its own directory, and
huggingface_hub resolves a relative HF_HOME against the current directory - so
with HF_HUB_OFFLINE=1 every Twitter simulation failed to find the model. These
tests pin the fix in both places that load .env.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
SCRIPTS = BACKEND / "scripts"
sys.path.insert(0, str(SCRIPTS))

import env_paths  # noqa: E402

CASES = [
    # (value in .env, expected result, or None for "left as it is")
    ("./data/hf-cache", "{root}/data/hf-cache"),
    ("data/hf-cache", "{root}/data/hf-cache"),
    ("./data/../data/hf-cache", "{root}/data/hf-cache"),
    ("/srv/hf", None),
    ("~/hf", None),
    ("", None),
]


@pytest.mark.parametrize("value,expected", CASES)
def test_relative_paths_are_pinned_to_the_root(tmp_path, value, expected):
    env = {"HF_HOME": value}
    changed = env_paths.pin_to_root(str(tmp_path), environ=env)
    if expected is None:
        assert env["HF_HOME"] == value
        assert changed == {}
    else:
        want = expected.format(root=tmp_path)
        assert env["HF_HOME"] == want
        assert changed == {"HF_HOME": want}


def test_an_unset_key_stays_unset(tmp_path):
    env = {}
    assert env_paths.pin_to_root(str(tmp_path), environ=env) == {}
    assert "HF_HOME" not in env


def test_backend_config_resolves_exactly_like_the_scripts(tmp_path):
    # app.config cannot import scripts/env_paths.py, so it carries a twin.
    from app.config import pin_root_relative_paths

    for value, _ in CASES:
        a, b = {"HF_HOME": value}, {"HF_HOME": value}
        env_paths.pin_to_root(str(tmp_path), environ=a)
        pin_root_relative_paths(str(tmp_path), environ=b)
        assert a == b, value


@pytest.mark.parametrize("script", [
    "run_parallel_simulation.py",
    "run_twitter_simulation.py",
    "run_reddit_simulation.py",
])
def test_each_simulation_script_pins_before_importing_oasis(script):
    # huggingface_hub reads HF_HOME when it is imported; oasis imports it.
    source = (SCRIPTS / script).read_text()
    pin = source.index("pin_to_root(_project_root)")
    assert pin > source.index("load_dotenv(")
    assert pin < source.index("import oasis")


def _hub_cache_from(cwd, prelude):
    """HF_HUB_CACHE as huggingface_hub sees it, from `cwd`, after `prelude`."""
    code = (
        f"import sys; sys.path.insert(0, {str(SCRIPTS)!r})\n"
        f"{prelude}\n"
        "import os\n"
        "from huggingface_hub import constants\n"
        "print(os.path.abspath(constants.HF_HUB_CACHE))\n"
    )
    env = dict(os.environ, HF_HOME="./data/hf-cache", HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_a_simulation_directory_finds_the_repo_cache(tmp_path):
    pytest.importorskip("huggingface_hub")
    root_cache = str(REPO / "data" / "hf-cache" / "hub")

    # The failure: from a simulation's own directory, a relative HF_HOME points
    # into that directory.
    unpinned = _hub_cache_from(tmp_path, "")
    assert unpinned == str(tmp_path / "data" / "hf-cache" / "hub")

    # The fix: pinned first, it points at the cache setup filled.
    pinned = _hub_cache_from(
        tmp_path, f"from env_paths import pin_to_root; pin_to_root({str(REPO)!r})")
    assert pinned == root_cache


def test_the_backend_pins_hf_home_for_the_simulations_it_starts():
    # Simulations inherit the backend's environment (simulation_runner copies
    # os.environ), so the backend's own value is what they get.
    code = "import os, app.config; print(os.environ['HF_HOME'])"
    env = dict(os.environ, HF_HOME="./data/hf-cache")
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env=env,
                         capture_output=True, text=True, check=True)
    assert os.path.isabs(out.stdout.strip())
    assert not out.stdout.strip().startswith(str(BACKEND) + os.sep)
