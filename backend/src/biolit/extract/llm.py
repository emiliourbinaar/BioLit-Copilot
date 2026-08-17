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

# The token counters read off `response.usage`, in the SDK's own names. ONE tuple, used both
# to seed the accumulator and to drive the accumulation, so a field cannot be counted but not
# reported or reported but never counted. `biolit_evals.extract_eval._diagnostics` builds
# every arm's log shape from this same tuple, which is what keeps the LLM arm and the
# deterministic controls from drifting into two shapes.
# NO PRICES HERE, DELIBERATELY. Per-token pricing changes and a rate committed to the repo
# would rot silently into a wrong cost estimate; the dollar conversion belongs in the eval
# report, next to the date it was true on.
USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


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
    BOTH CACHE COUNTERS ARE STILL RECORDED, precisely so the log PROVES that 0 rather than
    this docstring asserting it. They are the cheapest possible check on the paragraph above,
    and if a future prompt grows past 512 tokens they start moving on their own.

    `usage` IS ACCUMULATED FOR EVERY RESPONSE THE API RETURNS, INCLUDING THE ONES THAT YIELD
    NO FINDINGS, and is read BEFORE the stop-reason branch below for exactly that reason.
    The billing rules differ per stop reason -- a refusal declined before any output is not
    billed, a refusal declined mid-stream bills the streamed partial, and a `max_tokens`
    truncation bills everything it produced, thinking included -- so this class does not
    model them. It records what the API reports and lets the numbers say which case occurred.
    Dropping the usage of a non-`end_turn` response would silently under-report the cost of
    the failure mode that wastes the MOST money per paper: a truncation pays for up to
    `_MAX_TOKENS` of thinking and returns nothing scoreable.

    THE ERROR PATH IS THE ONE PLACE THIS COUNT IS A LOWER BOUND, and it is stated rather than
    left implicit. When `messages.create` raises there is no response object and therefore no
    `usage` to read: the SDK surfaces an exception, not a partial Message, so nothing can be
    attributed and nothing is added. Whether the server billed the attempt is unknowable from
    here. A run whose `errors` counter is non-zero therefore has a token total that
    UNDER-reports by an unknown amount, which is why the two are logged side by side and must
    be read together before a pilot's numbers are extrapolated to a full corpus.

    NO PRICES, IN EITHER DIRECTION -- see `USAGE_FIELDS` above. Tokens are recorded; the
    dollar conversion belongs in the eval report where it can carry the date it was true on.

    NO `fallbacks` PARAMETER, DELIBERATELY, against the general API guidance for
    claude-opus-5 code. This is an eval harness measuring one named model's extraction
    quality. A silent server-side substitution to a different model would corrupt the arm
    being measured while still reporting as a success -- exactly the silent-corruption class
    this task exists to prevent. A refusal must stay visible and counted. Do not "helpfully"
    add fallbacks here.
    """

    # `effort` DEFAULTS TO THE SETTING THAT WAS ACTUALLY RUN, not to the API's middle rung.
    # The full BC5CDR Test-500 arm this project quotes was run at `low`; `medium` has never
    # been run over a full corpus here. A default no measurement stands behind is how a rerun
    # silently produces numbers that are not comparable to the recorded ones, and the run log
    # would faithfully record the divergence only AFTER the money was spent.
    def __init__(self, client: Any, *, model: str = "claude-opus-5", effort: str = "low") -> None:
        # `model` and `effort` are PUBLIC because the run log must name which model and which
        # effort produced an arm, and the only value that cannot disagree with the one the API
        # was asked for is the one this object hands the API. A runner passing its own copy
        # alongside the extractor would log an unattributable run the moment the two diverge.
        self._client, self.model, self.effort = client, model, effort
        self.usage: Counter[str] = Counter(dict.fromkeys(USAGE_FIELDS, 0))
        self.refusals = 0
        self.unusable_stops: Counter[str] = Counter()
        self.out_of_range = 0
        self.licence_skipped = 0
        self.errors: Counter[str] = Counter()

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
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": _SCHEMA},
                },
                messages=[{"role": "user", "content": numbered}],
            )
        except Exception as exc:
            # A transport failure costs ONE PAPER, not the run: the eval makes 500 sequential
            # calls and runs at least twice, so a single 429 or dropped connection must not
            # discard everything before it. Same policy as the unusable stop reasons below,
            # and the catch is deliberately placed around THIS CALL ONLY so nothing downstream
            # of it -- a harness bug in span conversion, say -- is swallowed by it.
            self.errors[type(exc).__name__] += 1
            return []
        # ACCUMULATED BEFORE THE STOP-REASON BRANCH, so a response that yields no findings
        # still reports what it cost -- see the class docstring for why that is the whole
        # point. `usage` is a REQUIRED field on the SDK's `Message`, so it is present on
        # every response, refusals and truncations included, and is read without a guard.
        # `input_tokens` and `output_tokens` are required ints; both cache counters are
        # `Optional[int]` and are None whenever the API reports nothing, which for this arm
        # is always -- so None is counted as the zero it means, rather than raising.
        for field in USAGE_FIELDS:
            counted = getattr(response.usage, field)
            self.usage[field] += counted if counted is not None else 0
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
        try:
            indices = json.loads(payload)["finding_sentences"]
        except Exception as exc:
            # The stop reason said `end_turn` and the payload is still not the JSON the schema
            # promised. Structured outputs make this unlikely, not impossible, and mid-corpus
            # one JSONDecodeError costs every paper after it. Narrow on purpose: the
            # conversion below stays OUTSIDE the try, so a harness bug there raises instead of
            # being counted as a model failure.
            self.errors[type(exc).__name__] += 1
            return []
        out = findings_from_sentence_indices(text, indices)
        self.out_of_range += len(set(indices)) - len(out)
        return out
