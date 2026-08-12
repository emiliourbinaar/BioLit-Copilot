import json
from collections import Counter
from typing import Any

from biolit.domain.paper import Paper
from biolit.domain.records import Finding
from biolit.extract.base import findings_from_sentence_indices
from biolit.ner.windowing import sentence_spans

_SYSTEM = """You identify which sentences of a biomedical abstract state the study's findings.

A finding is a result the study reports: an observed effect, an adverse event, a measured
outcome. Background, methods, and motivation are not findings.

You will receive the abstract as a numbered list of sentences. Return the indices of the
sentences that state findings. Return an empty list if none do. Do not return an index that
is not in the list you were given."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"finding_sentences": {"type": "array", "items": {"type": "integer"}}},
    "required": ["finding_sentences"],
    "additionalProperties": False,
}

# max_tokens caps THINKING AND RESPONSE TEXT TOGETHER, and claude-opus-5 runs adaptive
# thinking by default (omitting the `thinking` parameter is not the same as disabling it).
# A small budget is therefore spent on thinking and returns stop_reason "max_tokens" with
# empty or partial content -- a JSONDecodeError mid-corpus, not a clean failure. 16000 is
# the documented non-streaming figure; the response itself is a handful of integers.
_MAX_TOKENS = 16000


class LlmExtractor:
    """Selects finding-bearing sentences with an LLM, by INDEX rather than by offset.

    The model never computes a character offset: it sees numbered sentences and returns
    indices, and the harness converts them via sentence_spans. That is what makes a
    fabricated span structurally impossible instead of merely detected.

    Note the schema gap: structured outputs cannot express numerical bounds, so it guarantees
    a list of integers but NOT that they are in range. findings_from_sentence_indices drops
    out-of-range values and `out_of_range` counts them as a reliability diagnostic. It counts
    INDICES DROPPED, not values emitted: that function dedupes (`sorted(set(indices))`), so
    [99, 99] is one dropped index. It is derived by subtracting what came back from the
    distinct indices asked for, rather than by restating the `0 <= i < len(spans)` predicate
    that already lives in base.py -- one predicate, one place, no drift.

    EVERY stop reason other than `end_turn` is counted and returns no findings, rather than
    only `refusal`. The documented set also holds max_tokens, stop_sequence, tool_use,
    pause_turn and model_context_window_exceeded, and on each of those the payload is absent
    or partial; letting them fall through to json.loads aborts the run mid-corpus. A
    truncated response is a COUNTED DIAGNOSTIC, not an exception -- an arm that raises loses
    the whole run, while a counter keeps the arm scoreable and the failure visible in the log.
    `refusals` stays its own field because a refusal is the failure mode the spec names;
    everything else lands in `unusable_stops`, keyed by reason, so a stop reason added to the
    API later is counted rather than parsed.

    `stop_details` is deliberately not read. It is populated ONLY when stop_reason is
    "refusal" and is null for every other stop reason, so any future reader who wants the
    refusal category must guard on that before touching it.

    NO `cache_control` MARKER, AND NO CACHING CLAIM. claude-opus-5's minimum cacheable
    prefix is 512 tokens; _SYSTEM is 452 ASCII characters, and no subword tokenizer emits
    more tokens than characters (every token covers at least one), so the prefix is provably
    under the minimum whatever the tokenizer. Below it, caching silently does
    not happen -- no error, just cache_creation_input_tokens: 0. An inert marker would read
    to the next maintainer as caching that occurs, so it is left out entirely.

    NO `fallbacks` PARAMETER, DELIBERATELY, against the general API guidance for
    claude-opus-5 code. This is an eval harness measuring one named model's extraction
    quality. A silent server-side substitution to a different model would corrupt the arm
    being measured while still reporting as a success -- exactly the silent-corruption class
    this task exists to prevent. A refusal must stay visible and counted. Do not "helpfully"
    add fallbacks here.
    """

    def __init__(
        self, client: Any, *, model: str = "claude-opus-5", effort: str = "medium"
    ) -> None:
        self._client, self._model, self._effort = client, model, effort
        self.refusals = 0
        self.unusable_stops: Counter[str] = Counter()
        self.out_of_range = 0
        self.licence_skipped = 0

    def findings(self, paper: Paper) -> list[Finding]:
        # Redundant with build_record's gate, deliberately: neither layer alone is
        # load-bearing for a licence decision. build_record is the enforcement point --
        # it suppresses the whole record, which this cannot do since it returns findings.
        if not paper.extraction_allowed:
            self.licence_skipped += 1
            return []
        text = paper.abstract or ""
        # The text is the only thing worth checking. sentence_spans NEVER returns an empty
        # list -- it falls back to [(0, len(text))] -- so a guard on `spans` can never fire.
        if not text.strip():
            return []
        spans = sentence_spans(text)
        numbered = "\n".join(f"[{i}] {text[start:end]}" for i, (start, end) in enumerate(spans))
        response = self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            output_config={
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            messages=[{"role": "user", "content": numbered}],
        )
        # Check stop_reason BEFORE reading content: on anything but end_turn, content is
        # empty or partial. A refusal empties key_findings only -- it says nothing about
        # entities, which come from the separate deterministic NER/linking stage.
        stop_reason = response.stop_reason
        if stop_reason != "end_turn":
            if stop_reason == "refusal":
                self.refusals += 1
            else:
                self.unusable_stops[str(stop_reason)] += 1
            return []
        # The first block is a thinking block on every live call, not the answer.
        payload = next((b.text for b in response.content if b.type == "text"), "")
        indices = json.loads(payload)["finding_sentences"]
        out = findings_from_sentence_indices(text, indices)
        self.out_of_range += len(set(indices)) - len(out)
        return out
