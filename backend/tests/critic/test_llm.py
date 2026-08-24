import json
from types import SimpleNamespace
from typing import Any

import pytest

from biolit.critic.base import CriticPair
from biolit.critic.llm import CRITIC_USAGE_FIELDS, CriticParseError, LlmCritic
from biolit.domain.records import ContradictionLabel
from biolit.extract.llm import USAGE_FIELDS

_DEFAULT_USAGE = {
    "input_tokens": 11,
    "output_tokens": 3,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

_PAIR = CriticPair("1", "2", "text a", "text b", "C1", "D1")


class _RecordingClient:
    """Fake anthropic-shaped client: records every system prompt, max_tokens, model, and
    output_config it was called with (so tests can assert what actually reached the client,
    not just what the code claims to send) and returns a configurable label -- or, via
    `raw_text`, a genuinely garbled payload that is not JSON at all -- optionally with
    configurable usage. A plain object -- never touches the network, never imports
    `anthropic`."""

    def __init__(
        self,
        *,
        label: str | None = None,
        raw_text: str | None = None,
        usage: dict[str, int] | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        if (label is None) == (raw_text is None):
            raise ValueError("pass exactly one of label or raw_text")
        self._label = label
        self._raw_text = raw_text
        self._usage = dict(_DEFAULT_USAGE) if usage is None else usage
        self._stop_reason = stop_reason
        self.systems: list[str] = []
        self.max_tokens_seen: list[int] = []
        self.models_seen: list[str] = []
        self.output_configs: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.systems.append(kwargs["system"])
        self.max_tokens_seen.append(kwargs["max_tokens"])
        self.models_seen.append(kwargs["model"])
        self.output_configs.append(kwargs["output_config"])
        text = (
            self._raw_text
            if self._raw_text is not None
            else json.dumps({"label": self._label, "rationale": "r"})
        )
        content = [SimpleNamespace(type="text", text=text)]
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            content=content,
            usage=SimpleNamespace(**self._usage),
        )


def test_the_same_prompt_serves_both_input_modes():
    """The arms are a DATA difference, not a code difference. Different prompts would
    confound extraction quality with prompt wording, and pricing biolit.extract at a real
    consumer is the entire point of the comparison."""
    client = _RecordingClient(label="agreement")
    critic = LlmCritic(client, model="claude-opus-5")
    critic.judge(CriticPair("1", "2", "full abstract a", "full abstract b", "C1", "D1"))
    critic.judge(CriticPair("3", "4", "finding a", "finding b", "C1", "D1"))
    assert client.systems[0] == client.systems[1]


def test_a_label_outside_the_three_raises_rather_than_defaulting():
    """A silent default scores as a wrong answer and vanishes from parse_failures --
    quieter, not louder (ADR-0014)."""
    with pytest.raises(CriticParseError, match="not_a_label"):
        LlmCritic(_RecordingClient(label="not_a_label"), model="m").judge(_PAIR)


def test_usage_accumulates_across_calls_over_every_usage_field():
    critic = LlmCritic(
        _RecordingClient(
            label="agreement",
            usage={
                "input_tokens": 11,
                "output_tokens": 3,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        ),
        model="m",
    )
    for _ in range(7):
        critic.judge(_PAIR)
    assert critic.usage["input_tokens"] == 77
    assert critic.usage["output_tokens"] == 21


def test_a_refusal_increments_refusals_and_returns_an_honest_insufficient_overlap_finding():
    """`judge()` must always return a ContradictionFinding -- the protocol has no room for
    'declined to answer' -- so a refusal is signalled via the `refusals` counter (diffed by
    the runner around each call, same convention as LlmExtractor) while still returning a
    real, honestly-labelled finding naming the refusal in its rationale."""
    critic = LlmCritic(_RecordingClient(label="agreement", stop_reason="refusal"), model="m")
    finding = critic.judge(_PAIR)
    assert critic.refusals == 1
    assert finding.label is ContradictionLabel.insufficient_overlap
    assert "refus" in finding.rationale.lower()


def test_critic_usage_fields_is_the_same_tuple_as_the_extractors_not_a_redefinition():
    """A second, independently-defined tuple of the same four names would be a second source
    of truth `run_critic_eval` and the extractor's own diagnostics could silently drift apart
    from -- this asserts reuse, not just equal contents."""
    assert CRITIC_USAGE_FIELDS is USAGE_FIELDS


def test_max_tokens_16000_actually_reaches_the_client():
    """max_tokens caps thinking AND response text together; too small a budget returns
    stop_reason 'max_tokens' with partial content -- a decode error mid-corpus, during a paid
    run. The constant exists to prevent exactly that, so what is actually sent must be pinned,
    not just declared in a comment a future refactor could silently drop or shrink."""
    client = _RecordingClient(label="agreement")
    LlmCritic(client, model="m").judge(_PAIR)
    assert client.max_tokens_seen[0] == 16000


def test_outbound_schema_constrains_label_to_exactly_the_three_and_requires_rationale():
    """The fake client echoes back whatever a test configured regardless of what schema was
    sent, so a mutant that widened the label enum, dropped `rationale` from `required`, or
    stopped constraining the shape at all would survive undetected unless the SENT schema is
    inspected directly -- the parse guard downstream must not be doing all the work alone."""
    client = _RecordingClient(label="agreement")
    LlmCritic(client, model="m").judge(_PAIR)
    schema = client.output_configs[0]["format"]["schema"]
    assert set(schema["properties"]["label"]["enum"]) == {
        "agreement",
        "contradiction",
        "insufficient_overlap",
    }
    assert "rationale" in schema["required"]


def test_a_genuinely_unparseable_payload_also_raises_criticparseerror():
    """'Unparseable response' names two distinct failure modes: an out-of-vocabulary label in
    an otherwise-valid payload (covered separately), and a payload that is not even valid
    JSON -- e.g. `json.loads` itself raising. A different code path to the same typed error,
    so it needs its own test rather than being assumed covered by the label test."""
    with pytest.raises(CriticParseError):
        LlmCritic(_RecordingClient(raw_text="not json at all {{{"), model="m").judge(_PAIR)


def test_no_parse_failures_counter_and_model_and_effort_reach_the_client():
    """A binding ruling, not an implementation detail: `run_critic_eval`'s own try/except
    around `judge()` is the SINGLE source of truth for a parse failure (module docstring), so
    a second `parse_failures` counter on the critic itself would be a second source of truth
    for one event. Pinned against a future regression that quietly adds one back. `model` and
    `effort` are asserted here too -- the same client-recording change makes it free, and the
    run log must be able to trust that what it names is what was actually sent."""
    client = _RecordingClient(label="agreement")
    critic = LlmCritic(client, model="claude-opus-5", effort="medium")
    critic.judge(_PAIR)
    assert not hasattr(critic, "parse_failures")
    assert client.models_seen[0] == "claude-opus-5"
    assert client.output_configs[0]["effort"] == "medium"
