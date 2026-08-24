"""The blind annotation sheet and the Gate 2 stop rule for the contradiction-detection eval.

The gold standard labels a pair `contradiction` when CTD records opposite evidence
directions for its two papers. That is a PROXY. Pi -- the fraction of such pairs a human
annotator agrees is a genuine disagreement -- is unknown until measured, and it can only be
measured if the annotator cannot see the gold label or the CTD directions while judging: a
leaked label makes pi unmeasurable, and nothing downstream could detect that it had
happened. `export_blind_sheet` is the whole protocol for keeping that from occurring, per
the ADR-0006 `domain_sample` blind-annotation precedent. `evaluate_gate2` is the stop rule
that decides, from the first annotated batch, whether continued spending on the proxy is
justified.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_gold import GoldPair
from biolit_evals.critic_scoring import wilson_interval


def _split_evenly(total: int, buckets: int) -> list[int]:
    """Split `total` into `buckets` near-equal shares; the remainder goes to the earliest
    buckets (in the caller's iteration order) so the split is deterministic."""
    base, remainder = divmod(total, buckets)
    return [base + (1 if i < remainder else 0) for i in range(buckets)]


def export_blind_sheet(
    pairs: Sequence[GoldPair],
    abstracts: Mapping[str, str],
    *,
    n_contradiction: int,
    n_other: int,
    rng: random.Random,
) -> list[dict]:
    """Build the annotator-facing sheet: `n_contradiction` contradiction pairs plus
    `n_other` pairs spread as evenly as possible across whichever non-contradiction labels
    are present in `pairs` (agreement, insufficient_overlap), with the remainder -- if
    `n_other` does not divide evenly -- going to the alphabetically-earlier label.

    Selection within each label takes pairs in the order they appear in `pairs`, so it is
    reproducible from that order alone; the ONE random step is the final shuffle, which
    exists purely so an annotator cannot infer a row's class from its position on the sheet.

    Each row carries exactly `{"pair_id", "chemical_id", "disease_id", "abstract_a",
    "abstract_b"}` -- no `label`, no `direction_a`/`direction_b`. That is the entire blind
    protocol: nothing in the row lets the annotator recover the gold answer they are meant
    to be checking.
    """
    by_label: dict[ContradictionLabel, list[GoldPair]] = {}
    for pair in pairs:
        by_label.setdefault(pair.label, []).append(pair)

    selected: list[GoldPair] = list(
        by_label.get(ContradictionLabel.contradiction, [])[:n_contradiction]
    )

    other_labels = sorted(label for label in by_label if label != ContradictionLabel.contradiction)
    if other_labels:
        quotas = _split_evenly(n_other, len(other_labels))
        for label, quota in zip(other_labels, quotas, strict=True):
            selected.extend(by_label[label][:quota])

    rng.shuffle(selected)

    return [
        {
            "pair_id": f"{pair.paper_id_a}_{pair.paper_id_b}",
            "chemical_id": pair.chemical_id,
            "disease_id": pair.disease_id,
            "abstract_a": abstracts[pair.paper_id_a],
            "abstract_b": abstracts[pair.paper_id_b],
        }
        for pair in selected
    ]


@dataclass(frozen=True)
class Gate2:
    """The result of `evaluate_gate2`. `interval` is a 95% Wilson interval over `genuine`
    of `n` -- reported so a reader can SEE how wide it is, not so the gate can act on it:
    see `evaluate_gate2` for why this is a stop rule and not a measurement of pi."""

    genuine: int
    n: int
    verdict: str
    interval: tuple[float, float]


def evaluate_gate2(genuine: int, n: int) -> Gate2:
    """The stop rule for the first 15-contradiction-pair annotation batch.

    | observed genuine `g` (of 15) | verdict            |
    |-------------------------------|--------------------|
    | g <= 7   (<= 50%)             | STOP               |
    | 8 <= g <= 10                  | CONTINUE_FLAGGED   |
    | g >= 11  (>= 73%)             | CONTINUE           |

    The cut sits at 7 because, under a true pi of 0.8, observing g <= 7 of 15 has
    probability 0.0042; under pi of 0.7 it is 0.0500. So it is a strong signal AGAINST pi
    >= 0.7 that fires rarely when the proxy is sound. It is deliberately weak against pi =
    0.6 (probability 0.2131): it is built to catch a proxy failing badly, not to flag one
    that is merely mediocre.

    IT IS A STOP RULE, NOT A MEASUREMENT OF PI. At n = 15, `wilson_interval(9, 15)` is
    (0.3575, 0.8018) -- width 0.4443 -- so a batch reading 0.6 cannot be distinguished from
    one reading 0.8. `interval` is reported precisely so that width is visible, not hidden
    behind a single point estimate; the verdict bands above are calibrated for this specific
    batch size (n = 15) and are not a general proportional rule for other n.
    """
    if genuine <= 7:
        verdict = "STOP"
    elif 8 <= genuine <= 10:
        verdict = "CONTINUE_FLAGGED"
    else:
        verdict = "CONTINUE"
    return Gate2(genuine=genuine, n=n, verdict=verdict, interval=wilson_interval(genuine, n))
