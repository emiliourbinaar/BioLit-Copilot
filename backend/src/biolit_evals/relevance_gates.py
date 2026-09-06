"""The pre-registered gates for the cluster-relevance annotation pass.

Design: `docs/superpowers/specs/2026-09-05-cluster-relevance-annotation-design.md` §5.
Written to match that section, which was fixed BEFORE any label existed. The bands here are
transcriptions of it, not choices made after seeing the data, and none of them may move to
match an observation.

Gate order is a dependency, not a presentation: Gate 1 decides whether the labels can be read
at all, Gate 2 decides whether a reading is attributable to the filter rather than to
annotator permissiveness, and only then do Gates 3 and 4 mean anything.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit_evals.annotation_export import parse_annotations
from biolit_evals.relevance_export import RELEVANCE_LABELS

#: §5 Gate 1. At or above this share of `cant_tell`, the schema does not fit the data.
CANT_TELL_LIMIT = 0.15

#: §5 Gate 2. Controls the annotator must place correctly for the reading to be attributable.
CONTROL_FLOOR = 7


@dataclass(frozen=True)
class LabelledRow:
    row_id: str
    query: str
    cluster_key: str
    is_distractor: bool
    label: str
    reason: str


@dataclass(frozen=True)
class Gate1:
    n_cant_tell: int
    n: int
    verdict: str


@dataclass(frozen=True)
class Gate2:
    n_off_topic: int
    n: int
    verdict: str


def load_labels(sheet: str, manifest: Mapping[str, object]) -> list[LabelledRow]:
    """Join the annotated sheet to the manifest that pinned the row set.

    REFUSES on any mismatch in either direction. A sheet row the manifest does not know means
    the sheet was regenerated or edited after labelling began, so every downstream count would
    be computed over a row set nobody froze. A manifest row missing from the sheet is the
    more dangerous mirror: a silently skipped row shrinks a gate denominator without changing
    any visible verdict, which is exactly the failure `parse_annotations`' strictness exists
    to prevent one level down.
    """
    annotations = parse_annotations(sheet, allowed=RELEVANCE_LABELS)
    known: Mapping[str, Mapping[str, object]] = manifest["rows"]  # type: ignore[assignment]

    unknown = sorted(set(annotations) - set(known))
    if unknown:
        raise ValueError(
            f"load_labels: sheet carries rows the manifest does not know: {unknown}. "
            "The sheet was edited or regenerated after labelling began; the frozen row set "
            "and the labels no longer describe the same pass."
        )
    missing = sorted(set(known) - set(annotations))
    if missing:
        raise ValueError(
            f"load_labels: manifest rows absent from the sheet: {missing}. A skipped row "
            "shrinks a gate denominator without changing any visible verdict."
        )

    return [
        LabelledRow(
            row_id=row_id,
            query=str(known[row_id]["query"]),
            cluster_key=str(known[row_id]["cluster_key"]),
            is_distractor=bool(known[row_id]["is_distractor"]),
            label=annotations[row_id].label,
            reason=annotations[row_id].reason,
        )
        for row_id in sorted(annotations)
    ]


def gate1_tractability(rows: Sequence[LabelledRow]) -> Gate1:
    """Can these labels be read at all, or is `cant_tell` swallowing the pass?

    REAL ROWS ONLY. The controls are a separate instrument with their own verdict, and they
    are built to be easy -- folding them in would dilute the denominator with rows designed
    not to be ambiguous.

    A `REVISE_SCHEMA` reading stops the pass: §5 forbids computing the filter's numbers at
    all, because a membership reading over rows the annotator could not judge is not
    interpretable. Same posture as ADR-0018's Gate 1, which was a genuine precondition.
    """
    real = [row for row in rows if not row.is_distractor]
    cant_tell = sum(1 for row in real if row.label == "cant_tell")
    verdict = "REVISE_SCHEMA" if real and cant_tell / len(real) >= CANT_TELL_LIMIT else "TRACTABLE"
    return Gate1(n_cant_tell=cant_tell, n=len(real), verdict=verdict)


def gate2_controls(rows: Sequence[LabelledRow]) -> Gate2:
    """Is the annotator using `off_topic` for its meaning?

    ADR-0018's contribution to this project's method, and the reason its π̂ readings were
    attributable where Phase 5's were not: without a control, a permissive annotator and a
    good filter produce the same numbers. A `CONFOUNDED` reading does not stop the pass, but
    it strips Gate 3 and Gate 4 of any causal reading -- they become description.
    """
    controls = [row for row in rows if row.is_distractor]
    off_topic = sum(1 for row in controls if row.label == "off_topic")
    verdict = "DISCRIMINATING" if off_topic >= CONTROL_FLOOR else "CONFOUNDED"
    return Gate2(n_off_topic=off_topic, n=len(controls), verdict=verdict)


def labels_hash(rows: Sequence[LabelledRow]) -> str:
    """Content-addressed over the JUDGMENTS, stable across order.

    `rows_hash` already pins which rows were shown; this pins what was said about them, so a
    gate recomputed against edited labels cannot claim the hash of the labels it was
    pre-registered against. Reasons are excluded deliberately: they are for a human reader,
    and a typo fix in one must not invalidate a frozen reading.
    """
    payload = json.dumps(
        sorted(
            (row.row_id, row.query, row.cluster_key, row.is_distractor, row.label) for row in rows
        ),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


#: Tier order for ordering readings. `cant_tell` is not a grade and is excluded (§2).
_TIER = {"answers": 0, "background": 1, "off_topic": 2}


@dataclass(frozen=True)
class Gate3:
    matrix: dict[tuple[str, str], int]
    false_drops: list[LabelledRow]
    n_kept: int
    n_dropped: int


@dataclass(frozen=True)
class Gate4a:
    correct: list[str]
    wrong: list[str]
    no_answers_queries: list[str]


@dataclass(frozen=True)
class Gate4b:
    inversions: int
    comparable_pairs: int
    per_query: dict[str, tuple[int, int]]


def gate3_membership(rows: Sequence[LabelledRow], *, kept: set[str]) -> Gate3:
    """The filter's confusion matrix against the labels, plus every false drop BY NAME.

    §5 pre-registers the action rather than a threshold: a dropped `answers` cluster is a
    defect to diagnose to its cause, not a rate to tolerate. With 83 rows and 13 drops the
    denominator cannot resolve a rate anyway -- one unexpected case moves it eight points --
    so this design deliberately does not lean on one, and hands back the cases instead.
    """
    real = [row for row in rows if not row.is_distractor and row.label in _TIER]
    matrix: dict[tuple[str, str], int] = {}
    false_drops: list[LabelledRow] = []
    for row in real:
        side = "kept" if row.cluster_key in kept else "dropped"
        matrix[(side, row.label)] = matrix.get((side, row.label), 0) + 1
        if side == "dropped" and row.label == "answers":
            false_drops.append(row)
    return Gate3(
        matrix=matrix,
        false_drops=false_drops,
        n_kept=sum(1 for row in real if row.cluster_key in kept),
        n_dropped=sum(1 for row in real if row.cluster_key not in kept),
    )


def _labels_by_query(rows: Sequence[LabelledRow]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row.is_distractor and row.label in _TIER:
            out.setdefault(row.query, {})[row.cluster_key] = row.label
    return out


def gate4a_leads(rows: Sequence[LabelledRow], *, ranked: Mapping[str, Sequence[str]]) -> Gate4a:
    """Does each query lead with an `answers` cluster?

    §5's Gate 4c is applied FIRST, and it is what makes the question this simple: a query with
    no `answers` cluster at all is excluded and reported separately, because the cluster that
    would answer it does not exist -- a retrieval or entity-linking failure belonging in
    DEFECTS.md, not a ranking failure. Every query surviving 4c therefore HAS an `answers`
    cluster, so "the best tier available to this query" is always `answers` and the two
    formulations coincide. This was first written as a `min` over available tiers; that was
    dead generality, and dead generality in a gate is where a later reader misjudges what was
    measured.
    """
    by_query = _labels_by_query(rows)
    correct, wrong, no_answers = [], [], []
    for query, labels in sorted(by_query.items()):
        order = [key for key in ranked.get(query, ()) if key in labels]
        if not order:
            continue
        if not any(label == "answers" for label in labels.values()):
            no_answers.append(query)
            continue
        (correct if labels[order[0]] == "answers" else wrong).append(query)
    return Gate4a(correct=correct, wrong=wrong, no_answers_queries=no_answers)


def gate4b_inversions(
    rows: Sequence[LabelledRow], *, ranked: Mapping[str, Sequence[str]]
) -> Gate4b:
    """Pairs the ranker places in the opposite order to their labels.

    TIES ARE NOT INVERSIONS and are not counted in the denominator. ADR-0020's rule is that
    clusters the labels grade equally are correctly in any order, so scoring them would hold
    the ranker to a criterion the ADR explicitly declines to adopt.

    Reported as a raw count over comparable pairs, never as a bare rate: the pairs share
    clusters and queries, so they are not independent and no interval would be honest.
    """
    by_query = _labels_by_query(rows)
    per_query: dict[str, tuple[int, int]] = {}
    for query, labels in sorted(by_query.items()):
        order = [key for key in ranked.get(query, ()) if key in labels]
        inversions = comparable = 0
        for i, earlier in enumerate(order):
            for later in order[i + 1 :]:
                if labels[earlier] == labels[later]:
                    continue
                comparable += 1
                if _TIER[labels[earlier]] > _TIER[labels[later]]:
                    inversions += 1
        per_query[query] = (inversions, comparable)
    return Gate4b(
        inversions=sum(i for i, _ in per_query.values()),
        comparable_pairs=sum(c for _, c in per_query.values()),
        per_query=per_query,
    )
