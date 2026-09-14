"""The per-request budget every simulation hands to the LLM client.

All three run_*_simulation.py entry points talk to the same endpoint, so the
knobs that decide whether a request can finish in time live here rather than
being re-derived (or, as was the case for the single-platform scripts,
silently left at the library default) in each one.

Three things bound a request:

- the timeout, which a queued request's wait counts against;
- the retry count, because a retry re-enters the same queue and so multiplies
  load exactly when the server is already the bottleneck;
- the output cap, which is the one that actually decides whether a local
  reasoning model can answer at all.

The output cap matters most and was previously absent. camel-ai's default
config sends no max_tokens at all, and vLLM then lets a generation run to
--max-model-len minus the prompt - on the shipped settings, ~30k tokens. A
model that opens with a <think> block will happily spend that budget, and at
the per-stream decode rate of a fully batched local server it cannot finish
inside any sane timeout. Every agent request then times out, the round loop
absorbs the failures, and the run reports full rounds against zero actions.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

# camel-ai would otherwise default to a 180s timeout with 3 retries, i.e. 4
# attempts and a 720s ceiling per agent request. An unresponsive endpoint
# therefore burns that budget on every round, which is how a dead endpoint
# once cost a full 11-hour run without failing the run.
DEFAULT_MODEL_TIMEOUT = 180.0
DEFAULT_MODEL_MAX_RETRIES = 3

# An OASIS agent answers with one tool call carrying a short post, so a
# four-figure cap is generous for the answer itself. It is deliberately NOT
# generous enough to absorb an open-ended reasoning preamble: a model that
# needs more than this is thinking, not acting, and cutting it off turns a
# silent multi-hour timeout into an immediate finish_reason="length" that says
# so. Set SIM_MODEL_MAX_TOKENS=0 to restore the old unbounded behaviour.
DEFAULT_MODEL_MAX_TOKENS = 1024


def get_model_request_budget() -> Tuple[float, int]:
    """Read the per-request timeout and retry count from the environment.

    Returns:
        Tuple[float, int]: the timeout in seconds and the retry count.
    """
    try:
        timeout = float(os.environ.get("SIM_MODEL_TIMEOUT", DEFAULT_MODEL_TIMEOUT))
    except ValueError:
        timeout = DEFAULT_MODEL_TIMEOUT
    try:
        max_retries = int(
            os.environ.get("SIM_MODEL_MAX_RETRIES", DEFAULT_MODEL_MAX_RETRIES)
        )
    except ValueError:
        max_retries = DEFAULT_MODEL_MAX_RETRIES
    return max(1.0, timeout), max(0, max_retries)


def get_model_max_tokens() -> Optional[int]:
    """Read the cap on generated tokens.

    Returns:
        Optional[int]: the cap, or None when the generation is left unbounded.
    """
    raw = os.environ.get("SIM_MODEL_MAX_TOKENS", "")
    if raw == "":
        return DEFAULT_MODEL_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MODEL_MAX_TOKENS
    # 0 (or anything below it) means "leave it to the server".
    return value if value > 0 else None


def get_model_extra_body() -> Dict[str, Any]:
    """Read extra JSON merged into every chat completion request.

    This is the escape hatch for server-specific switches that have no place
    in the OpenAI schema - above all turning a hybrid model's reasoning mode
    off, which on a vLLM-served Qwen3 build is:

        SIM_MODEL_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}

    Returns:
        Dict[str, Any]: the extra body, empty when none is configured.
    """
    raw = os.environ.get("SIM_MODEL_EXTRA_BODY", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_model_config_dict() -> Dict[str, Any]:
    """Build the config dict camel-ai feeds to chat.completions.create().

    camel-ai passes this straight through as keyword arguments, so both
    max_tokens and extra_body reach the endpoint unaltered.

    Returns:
        Dict[str, Any]: the request configuration.
    """
    config: Dict[str, Any] = {}

    max_tokens = get_model_max_tokens()
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    extra_body = get_model_extra_body()
    if extra_body:
        config["extra_body"] = extra_body

    return config


def describe_budget() -> str:
    """Render the budget as one line for the run log.

    Returns:
        str: a human-readable summary of every knob in force.
    """
    timeout, max_retries = get_model_request_budget()
    max_tokens = get_model_max_tokens()
    extra_body = get_model_extra_body()
    parts = [
        f"timeout={timeout:.0f}s",
        f"max_retries={max_retries}",
        f"max_tokens={max_tokens if max_tokens is not None else 'unbounded'}",
    ]
    if extra_body:
        parts.append(f"extra_body={json.dumps(extra_body, separators=(',', ':'))}")
    return ", ".join(parts)
