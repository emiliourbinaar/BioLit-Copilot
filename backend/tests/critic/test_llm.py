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
    """Fake anthropic-shaped client: records every system prompt it was called with (so the
    one-prompt-two-modes test can assert two calls used identical prompts) and returns a
    configurable label, optionally with configurable usage. A plain object -- never touches the
    network, never imports `anthropic`."""

    def __init__(
        self,
        *,
        label: str,
        usage: dict[str, int] | None = None,
        stop_reason: str = "end_turn",
    ) -> None:
        self._label = label
        self._usage = dict(_DEFAULT_USAGE) if usage is None else usage
        self._stop_reason = stop_reason
        self.systems: list[str] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.systems.append(kwargs["system"])
        content = [
            SimpleNamespace(
                type="text",
                text=json.dumps({"label": self._label, "rationale": "r"}),
            )
        ]
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
