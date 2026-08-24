"""The LLM pair-judgment arm -- one prompt, run over whichever text mode the caller hands it.

`LlmCritic` is deliberately the SAME code path for both LLM input modes the eval compares: the
full-abstract arm and the extracted-findings arm differ ONLY in the `text_a`/`text_b` a
`CriticPair` carries, never in the prompt or the schema. A prompt written differently per mode
would confound extraction quality with prompt wording, and pricing `biolit.extract`'s findings
against a real consumer -- this arm -- is the entire reason the two-mode comparison exists. If
the two modes ever need different wording, that is a finding about the comparison, not a
reason to fork this class.

`client` IS TYPED `Any`, exactly as `LlmExtractor` does, and this module imports NOTHING from
`anthropic`. `biolit` therefore gains no hard dependency on the SDK from this arm -- ADR-0015's
recorded placement argument for keeping the paid arms structurally optional.
"""

import json
from collections import Counter
from typing import Any

from biolit.critic.base import CriticPair
from biolit.domain.records import ContradictionFinding, ContradictionLabel
from biolit.extract.llm import USAGE_FIELDS

# Re-exported, not redefined: `run_critic_eval` and any future caller read usage fields off
# `biolit.extract.llm.USAGE_FIELDS` today, and a second, independently-defined tuple here would
# be a second source of truth for the same four field names -- exactly the drift the extractor
# module's own docstring warns against for `extract_eval._diagnostics`. This name exists so a
# reader of `biolit.critic.llm` can find "which usage fields does this arm report" without
# already knowing to look in `biolit.extract.llm`.
CRITIC_USAGE_FIELDS = USAGE_FIELDS

# Names the prompt version a run log line used, so a prompt edit shows up in the log rather than
# only in git history. Bump this string whenever `_SYSTEM` changes in a way that could move a
# score.
PROMPT_VERSION = "critic-llm-v1"

_SYSTEM = """You judge whether two biomedical texts, each concerning the same chemical and \
disease, agree, contradict, or do not overlap enough to compare.

You will receive Text A and Text B. Each is either a full paper abstract or a set of extracted \
finding sentences from one -- the same judgment applies either way, and you should not assume \
which one you were given.

Label the pair with exactly one of:
- "agreement": both texts support the same relationship between the chemical and the disease \
(for example, both report a therapeutic effect, or both report an adverse or causal effect).
- "contradiction": the texts assert opposite relationships for the same chemical and disease \
(for example, one reports a therapeutic effect and the other an adverse or causal effect).
- "insufficient_overlap": at least one text does not take a clear position on the relationship \
between the chemical and the disease, so the pair cannot be judged as agreeing or contradicting.

Return a short rationale naming the textual evidence you used for your label."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [label.value for label in ContradictionLabel],
        },
        "rationale": {"type": "string"},
    },
    "required": ["label", "rationale"],
    "additionalProperties": False,
}

# Same rationale as `biolit.extract.llm._MAX_TOKENS`: max_tokens caps THINKING AND RESPONSE TEXT
# TOGETHER, and adaptive thinking runs by default whenever `thinking` is omitted. A small budget
# is therefore spent on thinking and returns stop_reason "max_tokens" with empty or partial
# content -- a decode error mid-corpus rather than a clean, countable failure. 16000 is the same
# documented non-streaming figure the extractor uses; this response is a label and one sentence.
_MAX_TOKENS = 16000


class CriticParseError(Exception):
    """Raised when a response cannot be turned into a `ContradictionFinding`.

    Covers both an unparseable payload (bad JSON, a missing key) and a label outside the three
    `ContradictionLabel` values. NEVER a silent default: a default would score as an ordinary
    wrong answer and vanish from the run log's `parse_failures` count -- quieter, not louder,
    which is the opposite of ADR-0014's standing preference. `run_critic_eval` wraps `judge()`
    in `try/except Exception` for exactly this type, so raising it costs one pair, not the run.
    """


def _user_content(pair: CriticPair) -> str:
    return (
        f"Chemical: {pair.chemical_id}\n"
        f"Disease: {pair.disease_id}\n\n"
        f"Text A:\n{pair.text_a}\n\n"
        f"Text B:\n{pair.text_b}"
    )


class LlmCritic:
    """Judges one `CriticPair` with a single LLM call, implementing `Critic`.

    Mirrors `LlmExtractor`'s established shape (`biolit/extract/llm.py`): `self.usage` is a
    `Counter` seeded from `USAGE_FIELDS` so every field is present even before a call, and
    `self.refusals` is a plain stateful counter, not derived from `usage` -- `run_critic_eval`
    reads it by diffing around each `judge()` call, the same convention the extractor's own
    caller uses.

    Deliberately NO `parse_failures` counter here. `run_critic_eval`'s module docstring records
    the reasoning: a second counter tracking the same event `CriticParseError` already signals
    would be a second source of truth for one fact. Raising is the complete contract.
    """

    def __init__(self, client: Any, *, model: str, effort: str = "low") -> None:
        # `model` and `effort` are PUBLIC for the same reason `LlmExtractor` exposes them: the
        # run log must name which model and effort produced an arm, and the only value that
        # cannot disagree with what was actually sent is the one this object used to send it.
        self._client, self.model, self.effort = client, model, effort
        self.usage: Counter[str] = Counter(dict.fromkeys(USAGE_FIELDS, 0))
        self.refusals = 0

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            messages=[{"role": "user", "content": _user_content(pair)}],
        )
        # Accumulated BEFORE the stop-reason branch, matching `LlmExtractor`: a refusal still
        # cost tokens and the log must reflect that. `usage` is a required field on every
        # response the SDK returns; both cache counters may report `None`, which is counted as
        # the zero it means rather than raising.
        for field in USAGE_FIELDS:
            counted = getattr(response.usage, field)
            self.usage[field] += counted if counted is not None else 0

        if response.stop_reason == "refusal":
            self.refusals += 1
            # The protocol has no room for "declined to answer" -- `judge()` must return a
            # `ContradictionFinding` regardless. `insufficient_overlap` is the honest label: it
            # is not a claim that the papers agree or disagree, and `run_critic_eval`'s dual
            # scoring discards this label for the "refusals wrong" figure anyway, so what is
            # returned here does not decide the score -- it only has to be real and honest.
            return ContradictionFinding(
                paper_id_a=pair.paper_id_a,
                paper_id_b=pair.paper_id_b,
                label=ContradictionLabel.insufficient_overlap,
                rationale="refused: the model declined to judge this pair",
            )

        # The first content block is a thinking block on a live call, not the answer.
        payload = next((block.text for block in response.content if block.type == "text"), "")
        try:
            parsed = json.loads(payload)
            label = ContradictionLabel(parsed["label"])
            rationale = str(parsed["rationale"])
        except Exception as exc:
            # Structured outputs make a malformed payload unlikely, not impossible, and a label
            # outside the three enum values is exactly what `ContradictionLabel(...)` raises on
            # -- both land here, as one typed, catchable failure per pair instead of an
            # unhandled exception that aborts the corpus.
            raise CriticParseError(f"could not parse critic response: {exc}") from exc

        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=label,
            rationale=rationale,
        )
