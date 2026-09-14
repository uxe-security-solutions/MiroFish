"""A simulation refuses to start against an endpoint that cannot serve it.

The production failure was a run that reported 17 rounds against zero recorded
actions over 7.7 hours. Every agent request had timed out, the round loop had
absorbed the failures one by one, and the start-up check had passed because it
asked the endpoint for one token of small talk - which a server can answer
while being unable to produce a tool call over an agent-sized prompt inside the
run's timeout.

So the bar the check applies is the one the round loop applies without saying
so: a tool call, produced in time. These tests pin each verdict, because each
one names a different knob.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import llm_budget
import llm_preflight


# --- helpers -----------------------------------------------------------------

def _tool_call(name="create_post"):
    return SimpleNamespace(function=SimpleNamespace(name=name))


def _answer(content=None, tool_calls=None, finish_reason="stop",
            prompt_tokens=1500, completion_tokens=40):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _clear(monkeypatch):
    for key in (
        "SIM_MODEL_TIMEOUT",
        "SIM_MODEL_MAX_RETRIES",
        "SIM_MODEL_MAX_TOKENS",
        "SIM_MODEL_EXTRA_BODY",
        "SIM_PREFLIGHT_TIMEOUT",
        "SIM_PREFLIGHT_PROMPT_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)


# --- the budget handed to every request --------------------------------------

def test_output_is_capped_by_default(monkeypatch):
    """An uncapped generation runs to the server's context limit.

    camel-ai sends no max_tokens of its own, so leaving this unset is what let
    one agent's reasoning block spend the whole timeout.
    """
    _clear(monkeypatch)
    assert llm_budget.get_model_max_tokens() == llm_budget.DEFAULT_MODEL_MAX_TOKENS
    assert llm_budget.get_model_config_dict()["max_tokens"] == (
        llm_budget.DEFAULT_MODEL_MAX_TOKENS
    )


def test_the_cap_can_be_lifted_deliberately(monkeypatch):
    """0 means "leave it to the server", and must not become max_tokens=0."""
    _clear(monkeypatch)
    monkeypatch.setenv("SIM_MODEL_MAX_TOKENS", "0")
    assert llm_budget.get_model_max_tokens() is None
    assert "max_tokens" not in llm_budget.get_model_config_dict()


def test_a_nonsense_cap_falls_back_rather_than_crashing_the_run(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SIM_MODEL_MAX_TOKENS", "lots")
    assert llm_budget.get_model_max_tokens() == llm_budget.DEFAULT_MODEL_MAX_TOKENS


def test_extra_body_reaches_the_request(monkeypatch):
    """This is how a hybrid model's reasoning mode is turned off."""
    _clear(monkeypatch)
    monkeypatch.setenv(
        "SIM_MODEL_EXTRA_BODY",
        '{"chat_template_kwargs":{"enable_thinking":false}}',
    )
    config = llm_budget.get_model_config_dict()
    assert config["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_malformed_extra_body_is_ignored_not_fatal(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SIM_MODEL_EXTRA_BODY", "{not json")
    assert llm_budget.get_model_extra_body() == {}


# --- the preflight request ---------------------------------------------------

def test_the_preflight_is_allowed_what_the_agents_are_allowed(monkeypatch):
    """Any shorter rejects endpoints the run tolerates; any longer passes ones it does not."""
    _clear(monkeypatch)
    monkeypatch.setenv("SIM_MODEL_TIMEOUT", "300")
    assert llm_preflight.get_preflight_timeout() == 300.0


def test_the_preflight_timeout_can_still_be_overridden(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SIM_MODEL_TIMEOUT", "300")
    monkeypatch.setenv("SIM_PREFLIGHT_TIMEOUT", "45")
    assert llm_preflight.get_preflight_timeout() == 45.0


def test_the_prompt_is_agent_sized(monkeypatch):
    """Prefill over a timeline is half of what makes an agent request expensive."""
    _clear(monkeypatch)
    monkeypatch.setenv("SIM_PREFLIGHT_PROMPT_TOKENS", "1500")
    messages = llm_preflight.build_preflight_messages()
    words = sum(len(m["content"].split()) for m in messages)
    assert words >= 1000, "a one-line prompt does not exercise prefill"


def test_the_request_offers_tools_that_map_to_recordable_actions():
    names = {t["function"]["name"] for t in llm_preflight.PREFLIGHT_TOOLS}
    assert {"create_post", "like_post", "do_nothing"} <= names


# --- the verdicts ------------------------------------------------------------

def test_a_tool_call_passes():
    ok, detail = llm_preflight.classify_answer(
        _answer(tool_calls=[_tool_call()], finish_reason="tool_calls"),
        elapsed=4.0,
        timeout=300.0,
    )
    assert ok
    assert "called create_post" in detail


def test_a_passing_answer_still_reports_how_close_it_ran_to_the_timeout():
    """An idle endpoint that barely makes it will not survive a full batch."""
    ok, detail = llm_preflight.classify_answer(
        _answer(tool_calls=[_tool_call()], finish_reason="tool_calls"),
        elapsed=280.0,
        timeout=300.0,
    )
    assert ok
    assert "93% of the timeout" in detail


def test_a_reasoning_block_that_eats_the_cap_names_the_reasoning_switch():
    ok, detail = llm_preflight.classify_answer(
        _answer(content="<think>" + "weighing it up " * 200,
                finish_reason="length", completion_tokens=1024),
        elapsed=120.0,
        timeout=300.0,
    )
    assert not ok
    assert "enable_thinking" in detail
    assert "reasoning-parser" in detail


def test_a_truncated_answer_without_reasoning_names_the_cap_instead():
    ok, detail = llm_preflight.classify_answer(
        _answer(content="I think the second post is ",
                finish_reason="length", completion_tokens=1024),
        elapsed=120.0,
        timeout=300.0,
    )
    assert not ok
    assert "SIM_MODEL_MAX_TOKENS" in detail
    assert "enable_thinking" not in detail


def test_prose_without_a_tool_call_fails_because_it_records_no_action():
    """This is a working endpoint that still produces an empty action log."""
    ok, detail = llm_preflight.classify_answer(
        _answer(content="I would probably like the second post."),
        elapsed=8.0,
        timeout=300.0,
    )
    assert not ok
    assert "tool-call-parser" in detail


def test_a_two_hundred_with_no_choices_fails():
    ok, detail = llm_preflight.classify_answer(
        SimpleNamespace(choices=[], usage=None), elapsed=1.0, timeout=300.0
    )
    assert not ok
    assert "no choices" in detail


def test_every_verdict_carries_the_numbers_behind_it():
    """Without the token counts and the rate, "it is slow" names no knob."""
    for response in (
        _answer(tool_calls=[_tool_call()], finish_reason="tool_calls"),
        _answer(content="prose only"),
        _answer(content="<think>...", finish_reason="length"),
    ):
        _, detail = llm_preflight.classify_answer(response, elapsed=10.0, timeout=300.0)
        assert "prompt=1500 tok" in detail
        assert "tok/s" in detail


def test_a_timeout_says_the_agents_will_hit_the_same_wall():
    class APITimeoutError(Exception):
        pass

    detail = llm_preflight.describe_timeout(
        APITimeoutError("Request timed out."), elapsed=300.4, timeout=300.0
    )
    assert "did not finish an agent-sized one within 300s" in detail


def test_a_refused_connection_is_named_as_itself_not_as_a_timeout():
    class APIConnectionError(Exception):
        pass

    detail = llm_preflight.describe_timeout(
        APIConnectionError("Connection error."), elapsed=0.1, timeout=300.0
    )
    assert "APIConnectionError" in detail
    assert "agent-sized" not in detail


# --- a check held to a stricter bar than the run it is vouching for ----------

def test_a_stricter_preflight_budget_does_not_claim_a_verdict_on_the_run():
    """SIM_PREFLIGHT_TIMEOUT shipped at 60, sized for a one-token ping.

    An .env of that vintage holds the agent-shaped check to a fifth of what the
    agents get, so "every agent request will hit the same wall" would be a claim
    this check did not test.
    """
    class APITimeoutError(Exception):
        pass

    detail = llm_preflight.describe_timeout(
        APITimeoutError("Request timed out."),
        elapsed=60.3,
        timeout=60.0,
        model_timeout=300.0,
    )
    assert "STRICTER than the run" in detail
    assert "Unset SIM_PREFLIGHT_TIMEOUT" in detail
    assert "Every agent request will hit the same wall" not in detail


def test_an_equal_budget_does_speak_for_the_run():
    class APITimeoutError(Exception):
        pass

    detail = llm_preflight.describe_timeout(
        APITimeoutError("Request timed out."),
        elapsed=300.4,
        timeout=300.0,
        model_timeout=300.0,
    )
    assert "Every agent request will hit the same wall" in detail
    assert "STRICTER" not in detail
