"""One agent-shaped request that decides whether an endpoint can serve a run.

A simulation makes no progress at all when the configured model does not answer
in the shape OASIS needs, but the round loop tolerates per-agent failures and so
keeps going, reporting a full round count against zero real actions. Checking
once up front turns that silent multi-hour failure into an immediate one.

The check deliberately imitates a real agent request rather than sending a token
of small talk. An endpoint that answers "ping" in a second proves only that the
process is alive; it says nothing about whether it can produce a tool call over
an agent-sized prompt inside the run's timeout, and that is the thing that
actually decides whether a simulation records any behaviour.

Nothing here imports camel-ai or OASIS, so the rules below can be exercised
directly.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from llm_budget import (
    get_model_extra_body,
    get_model_max_tokens,
    get_model_request_budget,
)

# Roughly the size of the timeline an OASIS agent observes. Prompt size is half
# of what makes an agent request expensive - the prefill has to chew through the
# timeline - so a preflight that skips it is not measuring the same thing.
DEFAULT_PREFLIGHT_PROMPT_TOKENS = 1500

# Padding is counted in words and a token is well under a word, so this
# over-counts the prompt slightly. Erring large is the right direction: a
# preflight that under-states the real prompt passes runs that then fail.
_WORDS_PER_TOKEN = 0.75

# The other half of what makes an agent request expensive. Tools are what a
# reasoning model writes an essay about before it answers, and a tool call is
# the only thing OASIS can turn into a recorded action.
PREFLIGHT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_post",
            "description": "Publish a new post to the platform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text of the post.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "like_post",
            "description": "Like an existing post.",
            "parameters": {
                "type": "object",
                "properties": {
                    "post_id": {
                        "type": "integer",
                        "description": "The post to like.",
                    },
                },
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "description": "Take no action this round.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def get_preflight_timeout() -> float:
    """Read how long the preflight completion may take.

    The default is the run's own per-request timeout. A representative request
    is allowed exactly as long as the agents themselves get: any shorter and the
    preflight rejects an endpoint the run would have tolerated, any longer and
    it passes one the run will not.

    Returns:
        float: the timeout in seconds.
    """
    model_timeout, _ = get_model_request_budget()
    raw = os.environ.get("SIM_PREFLIGHT_TIMEOUT", "")
    if raw == "":
        return model_timeout
    try:
        return max(1.0, float(raw))
    except ValueError:
        return model_timeout


def build_preflight_messages() -> List[Dict[str, str]]:
    """Build an agent-sized prompt for the preflight request.

    Returns:
        List[Dict[str, str]]: the system and user messages.
    """
    try:
        target_tokens = int(
            os.environ.get(
                "SIM_PREFLIGHT_PROMPT_TOKENS", DEFAULT_PREFLIGHT_PROMPT_TOKENS
            )
        )
    except ValueError:
        target_tokens = DEFAULT_PREFLIGHT_PROMPT_TOKENS
    target_tokens = max(0, target_tokens)

    system = (
        "You are a social media user taking part in a simulation. Read the "
        "timeline below and take exactly one action by calling one of the "
        "tools you have been given. Answer with a tool call and nothing else."
    )

    # A filler timeline. The content is irrelevant; the length is the point.
    post = (
        "Post {n} by user_{n}: the new transit schedule changes my commute "
        "and I am not sure yet whether that is for the better. "
    )
    filler_words = int(target_tokens * _WORDS_PER_TOKEN)
    timeline = []
    words = 0
    n = 0
    while words < filler_words:
        n += 1
        text = post.format(n=n)
        timeline.append(text)
        words += len(text.split())

    user = "Your timeline:\n" + "\n".join(timeline) + "\nTake one action now."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def describe_timeout(exc: Exception, elapsed: float, timeout: float) -> str:
    """Describe a request that never came back.

    Args:
        exc: The exception raised by the client.
        elapsed: Seconds spent before it was raised.
        timeout: The timeout in force.

    Returns:
        str: the failure description.
    """
    # Name the exception type: a timeout, a refused connection and a rejected
    # key each call for a different fix.
    detail = f"{type(exc).__name__} after {elapsed:.1f}s: {exc}"
    if "Timeout" in type(exc).__name__:
        detail += (
            f" - the endpoint accepted the request but did not finish an "
            f"agent-sized one within {timeout:.0f}s. Every agent request will "
            f"hit the same wall."
        )
    return detail


def classify_answer(
    response: Any,
    elapsed: float,
    timeout: float,
) -> Tuple[bool, str]:
    """Decide whether an answer is one a simulation could have used.

    The bar is the same one the round loop applies without saying so: only a
    tool call becomes a recorded action, and only an answer that arrives inside
    the timeout arrives at all.

    Args:
        response: The chat completion returned by the endpoint.
        elapsed: Seconds the request took.
        timeout: The timeout it ran against.

    Returns:
        Tuple[bool, str]: whether the endpoint answered usably, plus a
        description carrying the numbers behind the verdict.
    """
    # A 200 carrying no choices means the endpoint is reachable but is not
    # serving completions, which fails agents just as surely as a timeout.
    choices = getattr(response, "choices", None)
    if not choices:
        return False, f"responded in {elapsed:.1f}s but returned no choices"

    choice = choices[0]
    message = getattr(choice, "message", None)
    finish_reason = getattr(choice, "finish_reason", None)

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

    # Report the numbers whatever the verdict: they are what turns "it is slow"
    # into a decision about which knob to turn.
    stats = f"{elapsed:.1f}s"
    if prompt_tokens is not None:
        stats += f", prompt={prompt_tokens} tok"
    if completion_tokens is not None:
        stats += f", completion={completion_tokens} tok"
        if elapsed > 0:
            stats += f" ({completion_tokens / elapsed:.1f} tok/s)"
    if finish_reason:
        stats += f", finish_reason={finish_reason}"

    content = (getattr(message, "content", None) or "") if message else ""
    # A hybrid model left in reasoning mode is the usual reason an answer never
    # arrives: it spends the whole budget before reaching the tool call.
    thinking = "<think>" in content or "</think>" in content
    reasoning_fix = (
        "turn reasoning off with SIM_MODEL_EXTRA_BODY="
        '{"chat_template_kwargs":{"enable_thinking":false}}, or serve vLLM '
        "with a matching --reasoning-parser."
    )

    tool_calls = getattr(message, "tool_calls", None) if message else None

    if finish_reason == "length" and not tool_calls:
        hint = "the answer was cut off at the output cap before any tool call. "
        if thinking:
            hint += f"The model spent the cap on a reasoning block: {reasoning_fix}"
        else:
            hint += "Raise SIM_MODEL_MAX_TOKENS, or shorten what agents are asked."
        return False, f"{stats} - {hint}"

    if not tool_calls:
        hint = (
            "answered in prose but called no tool, so OASIS would record no "
            "action for this agent. "
        )
        if thinking:
            hint += f"The answer contains a reasoning block: {reasoning_fix}"
        else:
            hint += (
                "Check that vLLM runs with --enable-auto-tool-choice and a "
                "--tool-call-parser this model actually emits."
            )
        return False, f"{stats} - {hint}"

    # It works. Say how close to the timeout it ran, because a request that only
    # just made it on an idle endpoint will not survive a full batch.
    called = getattr(getattr(tool_calls[0], "function", None), "name", "a tool")
    note = f"called {called}"
    if timeout > 0 and elapsed > timeout * 0.5:
        note += (
            f" but used {elapsed / timeout:.0%} of the timeout on an otherwise "
            f"idle endpoint; under a full batch it will be slower"
        )
    return True, f"{stats} - {note}"


async def check_endpoint(
    api_key: str,
    base_url: Optional[str],
    model: str,
    log,
    label: str = "",
) -> Tuple[bool, str]:
    """Send one agent-shaped request and judge the answer.

    No retries, so one request's worth of latency is all it costs, but otherwise
    the same timeout, output cap and extra body the agents will run against.

    Args:
        api_key: The API key.
        base_url: The endpoint, or None for the OpenAI default.
        model: The model name to request.
        log: A callable taking one message string.
        label: A prefix identifying which configuration this is.

    Returns:
        Tuple[bool, str]: whether the endpoint answered usably, plus a
        description.
    """
    timeout = get_preflight_timeout()
    max_tokens = get_model_max_tokens()
    extra_body = get_model_extra_body()

    log(
        f"{label} preflight: model={model}, "
        f"base_url={base_url[:40] if base_url else 'default'}..., "
        f"timeout={timeout:.0f}s, "
        f"max_tokens={max_tokens if max_tokens is not None else 'unbounded'}"
    )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=0,
    )

    request: Dict[str, Any] = {
        "model": model,
        "messages": build_preflight_messages(),
        "tools": PREFLIGHT_TOOLS,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    if extra_body:
        request["extra_body"] = extra_body

    started = datetime.now()
    try:
        response = await client.chat.completions.create(**request)
    except Exception as exc:
        elapsed = (datetime.now() - started).total_seconds()
        return False, describe_timeout(exc, elapsed, timeout)
    finally:
        try:
            await client.close()
        except Exception:
            pass

    elapsed = (datetime.now() - started).total_seconds()
    return classify_answer(response, elapsed, timeout)
