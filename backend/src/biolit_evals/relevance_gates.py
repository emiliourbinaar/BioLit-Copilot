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
    verdict = (
        "REVISE_SCHEMA" if real and cant_tell / len(real) >= CANT_TELL_LIMIT else "TRACTABLE"
    )
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
        sorted((row.row_id, row.query, row.cluster_key, row.is_distractor, row.label)
               for row in rows),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
