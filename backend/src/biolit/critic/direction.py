"""The direction-decomposition arm -- labels each paper's stance independently, then composes.

`DirectionCritic.judge` makes TWO LLM calls per `CriticPair`, ONE PER PAPER: each call asks only
"what does this single text assert about the chemical-disease relationship?", never "do these
two texts agree?". The two per-paper `PaperDirection` labels are then combined with `compose`
(`biolit.critic.base`) into the pair's `ContradictionLabel` -- the SAME `compose` the free
direction-lexicon baseline uses (`biolit_evals.critic_baselines.DirectionLexiconCritic`), so the
two arms are comparable by construction rather than by two rules that happen to agree today.
`PaperDirection` and `compose` are IMPORTED from `biolit.critic.base`, not redefined here.

COST NOTE, stated correctly because it is easy to oversell: on a SAMPLED eval (this project's
gold pairs are drawn independently, `assert_papers_disjoint` guarantees no paper repeats across
pairs) this arm costs roughly 2x the pair arm (`biolit.critic.llm.LlmCritic`), NOT less, because
a sampled pair has two mostly-distinct papers -- two calls against the pair arm's one. The
k(k-1)/2 -> k saving this decomposition is often pitched for only materialises on REAL clusters
where papers are reused across many pairs, which a disjoint sample structurally is not. What
this arm buys on THIS eval is error localisation to a single paper (a wrong pair label can be
attributed to one paper's direction call, not the pair as an undifferentiated unit) and a
deployment-cost projection for a real, non-sampled corpus -- not a cheaper eval run.

`CriticParseError` is IMPORTED from `biolit.critic.llm`, not redefined: it is one typed
condition ("could not turn a response into a typed value"), and a second, independently-defined
exception for the same condition would be a second source of truth `run_critic_eval`'s
try/except would have to know about. Deliberately NO `parse_failures` counter here either, for
the identical reason `LlmCritic` has none (see that module's docstring) -- raising is the
complete contract.

`client` IS TYPED `Any`, exactly as `LlmCritic` and `LlmExtractor` do, and this module imports
NOTHING from `anthropic`.
"""

import json
from collections import Counter
from typing import Any

from biolit.critic.base import CriticPair, PaperDirection, compose
from biolit.critic.llm import CriticParseError
from biolit.domain.records import ContradictionFinding, ContradictionLabel
from biolit.extract.llm import USAGE_FIELDS

_SYSTEM = """You judge the relationship a single biomedical text asserts between a chemical and \
a disease.

You will receive ONE text -- either a full paper abstract or a set of extracted finding \
sentences from one -- concerning one chemical and one disease. Judge only what THIS text \
asserts. You are not comparing it to any other text.

Label it with exactly one of:
- "causes": the text reports the chemical causing, inducing, or being associated with an \
adverse or causal effect on the disease.
- "treats": the text reports the chemical treating, ameliorating, or having a therapeutic \
effect on the disease.
- "neither": the text does not take a clear position on the relationship between the chemical \
and the disease.

Return a short rationale naming the textual evidence you used for your label."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": [direction.value for direction in PaperDirection],
        },
        "rationale": {"type": "string"},
    },
    "required": ["direction", "rationale"],
    "additionalProperties": False,
}

# Same rationale as `LlmCritic._MAX_TOKENS` (biolit/critic/llm.py): max_tokens caps thinking AND
# response text together, and a budget spent entirely on thinking returns stop_reason
# "max_tokens" with empty or partial content -- a decode error mid-corpus rather than a clean,
# countable failure. This call's response is smaller than the pair arm's (one word, one
# sentence), but the budget is the same documented figure for the same reason.
_MAX_TOKENS = 16000


def _user_content(text: str, chemical_id: str | None, disease_id: str | None) -> str:
    return f"Chemical: {chemical_id}\nDisease: {disease_id}\n\nText:\n{text}"


class DirectionCritic:
    """Judges one `CriticPair` with two LLM calls -- one per paper -- then composes, implementing
    `Critic`.

    Mirrors `LlmCritic`'s established shape: `self.usage` is a `Counter` seeded from
    `USAGE_FIELDS` so every field is present even before a call, accumulated over BOTH calls a
    `judge()` makes; `self.refusals` is a plain stateful counter (incremented once per REFUSED
    CALL, so a pair where both papers refuse counts 2, not 1), read by `run_critic_eval` by
    diffing around each `judge()` call, same convention as `LlmCritic`. Deliberately NO
    `parse_failures` counter, for the same reason `LlmCritic` has none.

    CACHE, keyed by `paper_id`: `assert_papers_disjoint` (the eval's own halting anchor) forbids
    any paper id from appearing in more than one pair, counting `paper_id_a` and `paper_id_b`
    together -- so it ALSO forbids `paper_id_a == paper_id_b` within a single pair, since that
    alone would already make one id appear twice. Under `run_critic_eval`, the only caller this
    arm has, a cache keyed by `paper_id` can therefore NEVER register a hit: not across pairs
    (forbidden across the whole corpus) and not within one pair (forbidden by the same anchor
    applied to one pair's own two ids). It is kept anyway, scoped as a plain instance dict, but
    every lookup ASSERTS the invariant it depends on -- that a repeated `paper_id` always carries
    the same text -- rather than silently trusting it, so a caller that bypasses the anchor (a
    hand-built `CriticPair` in a test, or a future caller of this class outside the eval runner)
    fails loudly instead of silently reusing a direction computed for different text.
    """

    def __init__(self, client: Any, *, model: str, effort: str = "low") -> None:
        self._client, self.model, self.effort = client, model, effort
        self.usage: Counter[str] = Counter(dict.fromkeys(USAGE_FIELDS, 0))
        self.refusals = 0
        self._cache: dict[str, tuple[str, PaperDirection]] = {}

    def judge(self, pair: CriticPair) -> ContradictionFinding:
        direction_a = self._direction_for(pair.paper_id_a, pair.text_a, pair)
        direction_b = self._direction_for(pair.paper_id_b, pair.text_b, pair)
        label: ContradictionLabel = compose(direction_a, direction_b)
        return ContradictionFinding(
            paper_id_a=pair.paper_id_a,
            paper_id_b=pair.paper_id_b,
            label=label,
            rationale=f"direction decomposition: a={direction_a.value}, b={direction_b.value}",
        )

    def _direction_for(self, paper_id: str, text: str, pair: CriticPair) -> PaperDirection:
        if paper_id in self._cache:
            cached_text, cached_direction = self._cache[paper_id]
            assert cached_text == text, (
                f"DirectionCritic cache: paper_id {paper_id!r} seen twice with different text. "
                "Papers are supposed to be disjoint across pairs (assert_papers_disjoint's own "
                "invariant) -- this critic will not silently reuse a direction computed for "
                "different text under the same id."
            )
            return cached_direction

        response = self._client.messages.create(
            model=self.model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": _SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": _user_content(text, pair.chemical_id, pair.disease_id),
                }
            ],
        )
        # Accumulated BEFORE the stop-reason branch, matching `LlmCritic`: a refusal still cost
        # tokens and the log must reflect that.
        for field in USAGE_FIELDS:
            counted = getattr(response.usage, field)
            self.usage[field] += counted if counted is not None else 0

        if response.stop_reason == "refusal":
            self.refusals += 1
            # `neither` is the honest per-paper label for "no position determined" -- not a
            # claim the paper takes no position, but a record that this call could not say what
            # it does. `compose` already treats `neither` as "cannot be compared", which is
            # exactly what a refused call is.
            direction = PaperDirection.neither
            self._cache[paper_id] = (text, direction)
            return direction

        # The first content block is a thinking block on a live call, not the answer.
        payload = next((block.text for block in response.content if block.type == "text"), "")
        try:
            parsed = json.loads(payload)
            direction = PaperDirection(parsed["direction"])
        except Exception as exc:
            raise CriticParseError(f"could not parse direction response: {exc}") from exc

        self._cache[paper_id] = (text, direction)
        return direction
