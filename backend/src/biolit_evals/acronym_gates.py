"""The pre-registered gates for the DEF-0001 acronym adjudication.

Design: `docs/superpowers/specs/2026-09-06-acronym-adjudication-design.md` §5, fixed before
any label existed. The bands here are transcriptions of it, not choices made after seeing the
data, and none of them may move to match an observation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit_evals.acronym_export import ACRONYM_LABELS
from biolit_evals.annotation_export import parse_annotations

#: §5 Gate 1. At or above this share of `cant_tell` on real rows the sheet is not showing
#: enough text per row.
#:
#: ⚠️ DELIBERATELY NOT the relevance pass's 0.15. ADR-0018 recorded the specific mistake of
#: carrying a constant calibrated for one question into a second: 0.15 was calibrated for "can
#: you judge topical relevance", and this asks "can you tell what this acronym means here",
#: where nine of the 44 pairs carry no in-document gloss at all. 0.25 is the point at which the
#: deliverable stops being computable -- below it the rate still rests on at least 33 pairs.
CANT_TELL_LIMIT = 0.25


@dataclass(frozen=True)
class LabelledPair:
    row_id: str
    surface: str
    concept_id: str
    n_mentions: int
    type_violation: bool
    is_control: bool
    label: str
    reason: str


@dataclass(frozen=True)
class Gate1:
    n_cant_tell: int
    n: int
    verdict: str


def gate1_tractability(rows: Sequence[LabelledPair]) -> Gate1:
    """Is the sheet showing enough text to judge, or is `cant_tell` swallowing the pass?

    REAL ROWS ONLY. The controls are a separate instrument with their own verdict and are
    built to be unambiguous; folding them in would dilute the denominator with rows designed
    to be easy.

    A `REVISE_CONTEXT` reading does NOT stop the pass -- unlike the relevance pass's Gate 1,
    which was a precondition. It says the instrument is wrong rather than the finding, and the
    action is to widen the context windows and re-label. `cant_tell` rows are excluded from
    Gate 3's denominator either way, and the exclusion is reported with the rate.
    """
    real = [row for row in rows if not row.is_control]
    cant_tell = sum(1 for row in real if row.label == "cant_tell")
    verdict = "REVISE_CONTEXT" if real and cant_tell / len(real) >= CANT_TELL_LIMIT else "TRACTABLE"
    return Gate1(n_cant_tell=cant_tell, n=len(real), verdict=verdict)


#: §5 Gate 2. Controls the annotator must reject for the reading to be attributable. Carried
#: from the relevance pass, and the carry is justified rather than assumed: unlike Gate 1's
#: threshold this is the same instrument asking the same question -- is the rejection label
#: being used for its meaning -- over controls built the same way to be unambiguous.
CONTROL_FLOOR = 7


@dataclass(frozen=True)
class Gate2:
    n_wrong: int
    n: int
    verdict: str


def gate2_controls(rows: Sequence[LabelledPair]) -> Gate2:
    """Is the annotator rejecting links that are plainly not the concept?

    ⚠️ ONLY `wrong` COUNTS. A control is a real surface shown against a concept belonging to a
    different pair -- not the right concept at the wrong level -- so `granularity` is not a
    rejection here, and accepting it would let a systematically hedging annotator clear the
    gate while telling us nothing about whether they can reject anything.

    A `CONFOUNDED` reading does not stop the pass; it strips Gate 3 of any causal reading and
    leaves it as description.
    """
    controls = [row for row in rows if row.is_control]
    rejected = sum(1 for row in controls if row.label == "wrong")
    verdict = "DISCRIMINATING" if rejected >= CONTROL_FLOOR else "CONFOUNDED"
    return Gate2(n_wrong=rejected, n=len(controls), verdict=verdict)


#: §5 Gate 3. `cant_tell` is not a grade and never enters a rate's denominator.
_GRADES = ("correct", "wrong", "granularity")


@dataclass(frozen=True)
class Gate3:
    by_pair: dict[str, int]
    by_mention: dict[str, int]
    n_pairs: int
    n_mentions: int
    n_cant_tell: int


def gate3_rate(rows: Sequence[LabelledPair]) -> Gate3:
    """The deliverable DEF-0001 asks for. No threshold and nothing to pass or fail.

    ⚠️ BOTH WEIGHTINGS, ALWAYS. Per pair answers "how often does this mechanism produce a wrong
    concept"; mention-weighted answers "how much wrong text does a reader actually see". They
    diverge sharply -- `APT` alone is 26 of the 204 mentions -- so quoting one for the other is
    a real misreading, not a rounding.

    ⛔ NO INTERVAL. §1: the 44 pairs are a CENSUS of this corpus, not a draw from a population,
    so there is no sampling distribution for an interval to describe. Anything computed here
    describes these eight queries.

    Counts across all three grades rather than a single "error" number, because collapsing
    `granularity` into `wrong` would file DEF-0002's failure shape as a plain mislink.
    """
    real = [row for row in rows if not row.is_control]
    graded = [row for row in real if row.label in _GRADES]
    return Gate3(
        by_pair={
            grade: sum(1 for row in graded if row.label == grade)
            for grade in _GRADES
            if any(row.label == grade for row in graded)
        },
        by_mention={
            grade: sum(row.n_mentions for row in graded if row.label == grade)
            for grade in _GRADES
            if any(row.label == grade for row in graded)
        },
        n_pairs=len(graded),
        n_mentions=sum(row.n_mentions for row in graded),
        n_cant_tell=sum(1 for row in real if row.label == "cant_tell"),
    )


@dataclass(frozen=True)
class Gate4:
    flagged: dict[str, int]
    unflagged: dict[str, int]


def gate4_type_violation(rows: Sequence[LabelledPair]) -> Gate4:
    """DEF-0004's flag against the labels. ⚠️ DESCRIPTIVE ONLY -- counts and nothing else.

    ⛔ NO TEST STATISTIC, NO RATE, NO LIFT, AND THE ABSENCE IS THE POINT. §7.1: all ten flagged
    pairs were named to the annotator in the session that produced this design, so their labels
    cannot be the blind test of the flag this gate was built to be. Any summary statistic here
    would read as evidence for the screening signal when the honest reading is only "this is
    what the labels said about rows the annotator already knew were flagged".

    DEF-0004 rests on the mechanical inconsistency, which needs no labels at all. What these
    counts CAN do is say how many flagged pairs turned out to be link errors rather than NER
    errors -- a decomposition, not a validation.
    """

    def tally(subset: Sequence[LabelledPair]) -> dict[str, int]:
        return {
            grade: sum(1 for row in subset if row.label == grade)
            for grade in _GRADES
            if any(row.label == grade for row in subset)
        }

    real = [row for row in rows if not row.is_control]
    return Gate4(
        flagged=tally([row for row in real if row.type_violation]),
        unflagged=tally([row for row in real if not row.type_violation]),
    )


#: §7.1. The 16 surfaces named to the annotator BEFORE labelling began. Nine came from
#: DEFECTS.md with a direction attached -- GSH, ATN, CP, CPA, AITC as wrong; ICH, ATP, HCC, FXS
#: as correct -- and the ten type-violating pairs were shown as a table in the session that
#: produced this design.
#:
#: ⛔ FIXED HERE, NOT RECOMPUTED. Deriving the split later from "which pairs does the code
#: still know we mentioned" would drift with whatever happened to remain visible, which is the
#: opposite of a pre-registration. If another pair is disclosed before labelling, it is added
#: here and the addition is a visible change to the record.
DISCLOSED_SURFACES = frozenset(
    {
        # DEFECTS.md's table, with a direction attached
        "GSH", "ATN", "CP", "CPA", "AITC", "ICH", "ATP", "HCC", "FXS",
        # DEF-0004's type-violation table, disclosed as flagged but not as an answer
        "APT", "RA", "AT", "PCC", "BLM", "CD", "DIC",
    }
)  # fmt: skip


def split_by_disclosure(
    rows: Sequence[LabelledPair],
) -> tuple[list[LabelledPair], list[LabelledPair]]:
    """Partition the real rows into (disclosed, undisclosed) per §7.1.

    Same treatment §8.2 of the relevance design gave contaminated queries: reported separately
    as "not blind", never folded into a denominator that reads clean. Controls belong to
    neither side -- they are a different instrument with their own verdict.
    """
    real = [row for row in rows if not row.is_control]
    disclosed = [row for row in real if row.surface in DISCLOSED_SURFACES]
    undisclosed = [row for row in real if row.surface not in DISCLOSED_SURFACES]
    return disclosed, undisclosed


def load_labels(sheet: str, manifest: Mapping[str, object]) -> list[LabelledPair]:
    """Join the annotated sheet to the manifest that pinned the row set.

    REFUSES on any mismatch in either direction. A sheet row the manifest does not know means
    the sheet was edited or regenerated after labelling began, so the frozen row set and the
    labels no longer describe the same pass. A manifest row missing from the sheet is the more
    dangerous mirror: a silently skipped row shrinks a gate denominator without changing any
    visible verdict.

    ⚠️ `n_mentions` and `type_violation` come from the MANIFEST, never from the sheet -- the
    sheet deliberately never carried either, and a version that did would mean the pass was not
    blind.
    """
    annotations = parse_annotations(sheet, allowed=ACRONYM_LABELS)
    known: Mapping[str, Mapping[str, object]] = manifest["rows"]  # type: ignore[assignment]

    unknown = sorted(set(annotations) - set(known))
    if unknown:
        raise ValueError(
            f"load_labels: sheet carries rows the manifest does not know: {unknown}. The sheet "
            "was edited or regenerated after labelling began; the frozen row set and the "
            "labels no longer describe the same pass."
        )
    missing = sorted(set(known) - set(annotations))
    if missing:
        raise ValueError(
            f"load_labels: manifest rows absent from the sheet: {missing}. A skipped row "
            "shrinks a gate denominator without changing any visible verdict."
        )

    return [
        LabelledPair(
            row_id=row_id,
            surface=str(known[row_id]["surface"]),
            concept_id=str(known[row_id]["concept_id"]),
            n_mentions=int(known[row_id]["n_mentions"]),  # type: ignore[arg-type]
            type_violation=bool(known[row_id]["type_violation"]),
            is_control=bool(known[row_id]["is_control"]),
            label=annotations[row_id].label,
            reason=annotations[row_id].reason,
        )
        for row_id in sorted(annotations)
    ]
