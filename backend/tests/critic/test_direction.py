import json
from types import SimpleNamespace
from typing import Any

import pytest

from biolit.critic.base import CriticPair
from biolit.critic.direction import CriticParseError, DirectionCritic
from biolit.critic.llm import CriticParseError as LlmCriticParseError
from biolit.domain.records import ContradictionLabel

_DEFAULT_USAGE = {
    "input_tokens": 5,
    "output_tokens": 2,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


class _RecordingClient:
    """Fake anthropic-shaped client for `DirectionCritic` tests. Each `.messages.create` call
    consumes the NEXT entry of `responses` in order -- a dict of `direction` (or `raw_text` for
    a genuinely garbled, non-JSON payload), plus optional `stop_reason` and `usage` -- so a test
    can make the first call (paper A) and the second call (paper B) behave differently. Records
    every system/model/max_tokens/output_config/user-content it was called with, so tests can
    assert what actually reached the client. A plain object -- never touches the network, never
    imports `anthropic`."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.systems: list[str] = []
        self.models_seen: list[str] = []
        self.max_tokens_seen: list[int] = []
        self.output_configs: list[dict] = []
        self.user_contents: list[str] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        call_index = len(self.systems)
        self.systems.append(kwargs["system"])
        self.models_seen.append(kwargs["model"])
        self.max_tokens_seen.append(kwargs["max_tokens"])
        self.output_configs.append(kwargs["output_config"])
        self.user_contents.append(kwargs["messages"][0]["content"])
        spec = self._responses[call_index]
        text = (
            spec["raw_text"]
            if "raw_text" in spec
            else json.dumps({"direction": spec["direction"], "rationale": "r"})
        )
        usage = spec.get("usage", _DEFAULT_USAGE)
        return SimpleNamespace(
            stop_reason=spec.get("stop_reason", "end_turn"),
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(**usage),
        )


def test_one_call_per_paper_is_made_not_one_per_pair():
    client = _RecordingClient([{"direction": "causes"}, {"direction": "causes"}])
    DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert len(client.systems) == 2


def test_judge_composes_the_two_per_paper_directions_into_the_pair_label():
    """The two calls answering `causes` and `treats` respectively must compose to
    `contradiction` via the SHARED `compose` -- pinning that `judge()` actually feeds the two
    LLM answers into `compose` rather than, say, always returning the first paper's direction."""
    client = _RecordingClient([{"direction": "causes"}, {"direction": "treats"}])
    finding = DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert finding.label is ContradictionLabel.contradiction


def test_usage_accumulates_over_both_calls_a_single_judge_makes():
    """Two calls per pair, and each contributes to `self.usage` -- pinning that accumulation
    happens for BOTH the paper-A and paper-B calls, not just the first."""
    client = _RecordingClient(
        [
            {"direction": "causes", "usage": {**_DEFAULT_USAGE, "input_tokens": 10}},
            {"direction": "treats", "usage": {**_DEFAULT_USAGE, "input_tokens": 4}},
        ]
    )
    critic = DirectionCritic(client, model="m")
    critic.judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert critic.usage["input_tokens"] == 14


def test_model_and_effort_reach_the_client_on_both_calls():
    client = _RecordingClient([{"direction": "causes"}, {"direction": "treats"}])
    DirectionCritic(client, model="claude-opus-5", effort="medium").judge(
        CriticPair("1", "2", "a", "b", "C1", "D1")
    )
    assert client.models_seen == ["claude-opus-5", "claude-opus-5"]
    assert client.output_configs[0]["effort"] == "medium"
    assert client.output_configs[1]["effort"] == "medium"


def test_a_refusal_on_one_paper_increments_refusals_and_labels_that_paper_neither():
    """A refused per-paper call cannot be trusted as a direction, so it is recorded as `neither`
    (honest: "no position determined") and `compose` correctly treats that as
    `insufficient_overlap` even though the OTHER paper answered normally -- `judge()` must still
    make both calls and return a real finding, never raise, on a refusal."""
    client = _RecordingClient(
        [{"direction": "causes", "stop_reason": "refusal"}, {"direction": "treats"}]
    )
    critic = DirectionCritic(client, model="m")
    finding = critic.judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert critic.refusals == 1
    assert len(client.systems) == 2
    assert finding.label is ContradictionLabel.insufficient_overlap


def test_a_direction_outside_the_three_raises_criticparseerror_rather_than_defaulting():
    """Same ADR-0014 preference `LlmCritic` records: a silent default would score as an ordinary
    wrong answer and vanish, quieter rather than louder."""
    client = _RecordingClient([{"direction": "not_a_direction"}, {"direction": "causes"}])
    with pytest.raises(CriticParseError, match="not_a_direction"):
        DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))


def test_criticparseerror_is_the_same_type_llmcritic_raises_not_a_redefinition():
    """A second, independently-defined exception for the same condition would be a second
    source of truth `run_critic_eval`'s try/except would have to know about -- this asserts
    reuse, not just an equal name."""
    assert CriticParseError is LlmCriticParseError


def test_no_parse_failures_counter():
    """A binding ruling, not an implementation detail (see the module docstring):
    `run_critic_eval`'s own try/except around `judge()` is the SINGLE source of truth for a
    parse failure, so a second counter on the critic itself would be a second source of truth
    for one event."""
    client = _RecordingClient([{"direction": "causes"}, {"direction": "treats"}])
    critic = DirectionCritic(client, model="m")
    critic.judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert not hasattr(critic, "parse_failures")


def test_cache_reuses_a_direction_when_the_same_paper_id_and_text_recur():
    """The cache's only legitimate hit scenario: the SAME paper_id with the SAME text seen
    again. `assert_papers_disjoint` forbids this from ever happening across pairs -- it even
    forbids `paper_id_a == paper_id_b` within one pair -- so this test exercises it directly via
    a hand-built degenerate `CriticPair`, the only way it can be observed at all."""
    client = _RecordingClient([{"direction": "causes"}])
    critic = DirectionCritic(client, model="m")
    finding = critic.judge(CriticPair("1", "1", "same text", "same text", "C1", "D1"))
    assert len(client.systems) == 1
    assert finding.label is ContradictionLabel.agreement


def test_cache_asserts_rather_than_silently_reuses_when_paper_id_recurs_with_different_text():
    """If a `paper_id` recurs (across two `judge()` calls, or on the same instance) with
    DIFFERENT text, the disjointness invariant the cache depends on has been violated -- this
    must fail loudly (ADR-0014's standing preference), not silently serve a direction computed
    for the wrong text."""
    client = _RecordingClient([{"direction": "causes"}, {"direction": "treats"}])
    critic = DirectionCritic(client, model="m")
    critic.judge(CriticPair("1", "2", "text-x", "b", "C1", "D1"))
    with pytest.raises(AssertionError, match="paper_id '1'"):
        critic.judge(CriticPair("1", "3", "text-y", "c", "C1", "D1"))


def test_max_tokens_16000_actually_reaches_the_client_on_both_calls():
    """Same rationale as `LlmCritic`'s equivalent test: what is actually SENT must be pinned,
    not just declared in a comment a future refactor could silently drop or shrink."""
    client = _RecordingClient([{"direction": "causes"}, {"direction": "treats"}])
    DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    assert client.max_tokens_seen == [16000, 16000]


def test_a_genuinely_unparseable_payload_also_raises_criticparseerror():
    """'Unparseable response' names two distinct failure modes: an out-of-vocabulary direction
    in an otherwise-valid payload (covered separately) and a payload that is not valid JSON at
    all -- e.g. `json.loads` itself raising. A different code path to the same typed error."""
    client = _RecordingClient([{"raw_text": "not json at all {{{"}])
    with pytest.raises(CriticParseError):
        DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))


def test_outbound_schema_constrains_direction_to_exactly_the_three_and_requires_rationale():
    """The fake client echoes back whatever a test configured regardless of what schema was
    sent, so a mutant that widened the direction enum, dropped `rationale` from `required`, or
    stopped constraining the shape at all would survive undetected unless the SENT schema is
    inspected directly."""
    client = _RecordingClient([{"direction": "causes"}, {"direction": "treats"}])
    DirectionCritic(client, model="m").judge(CriticPair("1", "2", "a", "b", "C1", "D1"))
    schema = client.output_configs[0]["format"]["schema"]
    assert set(schema["properties"]["direction"]["enum"]) == {"causes", "treats", "neither"}
    assert "rationale" in schema["required"]
