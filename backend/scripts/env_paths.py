"""Resolve the relative paths .env holds against the repository root.

.env writes some paths relative to the repository root - above all
HF_HOME=./data/hf-cache, where `provision_local.sh setup` puts the
Twitter/twhin-bert-base recommender model. But no process that reads it runs
from the root: the backend runs in backend/, and every simulation runs in its
own directory under backend/uploads/simulations/. huggingface_hub resolves a
relative HF_HOME against the current directory at the moment it opens a file, so
each of them looked for the model somewhere else, found nothing, and - with
HF_HUB_OFFLINE=1 - OASIS failed with "Failed to load the model:
Twitter/twhin-bert-base". Every Twitter and parallel simulation did.

Pinning the value to an absolute path as soon as .env is loaded, before
huggingface_hub is imported, fixes that for this process and for every child it
starts, since they inherit the environment.

backend/app/config.py does the same for the Flask process. It cannot import
this module (backend/scripts is not on the app's import path), so it carries a
twin of this function; backend/tests/test_env_paths.py holds both to the same
behaviour.
"""

import os
from typing import Dict, Iterable, MutableMapping, Optional

# Keys whose value .env writes as a filesystem path relative to the repo root.
ROOT_RELATIVE_KEYS = ("HF_HOME",)


def pin_to_root(
    root: str,
    keys: Iterable[str] = ROOT_RELATIVE_KEYS,
    environ: Optional[MutableMapping[str, str]] = None,
) -> Dict[str, str]:
    """Rewrite each relative path in `keys` as an absolute path under `root`.

    Absolute values, `~` paths and unset or empty keys are left alone. Returns
    the keys that were rewritten, with their new values.
    """
    env = os.environ if environ is None else environ
    changed = {}
    for key in keys:
        value = env.get(key)
        if not value:
            continue
        expanded = os.path.expanduser(os.path.expandvars(value))
        if os.path.isabs(expanded):
            continue
        env[key] = os.path.normpath(os.path.join(os.path.abspath(root), expanded))
        changed[key] = env[key]
    return changed
