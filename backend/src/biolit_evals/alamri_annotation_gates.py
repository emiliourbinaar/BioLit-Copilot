"""Read the annotated Alamri sheet and run the gates.

`evaluate_gate1`, `evaluate_gate2`, `wilson_interval` and `parse_annotations` are imported
UNMODIFIED from Phase 5. That is the design's central claim: the instrument that produced
ADR-0017's reading is the same instrument reading this batch, so the two are comparable and
neither was tuned to its own result.

What is new here is only the JOIN (annotator verdicts onto strata) and the two CONTROL reads
that a single annotator makes necessary -- the strictness read over the agreement pairs and
the discriminator read over the cross-question distractors (spec §5).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit_evals.alamri_gold import (
    STRATUM_AGREEMENT,
    STRATUM_CLEAN,
    STRATUM_DISTRACTOR,
    STRATUM_FLAGGED,
    AlamriPair,
)
from biolit_evals.annotation_export import Annotation

#: The annotator verdict that counts as `genuine` for Gate 2 -- pi is the fraction of derived
#: contradiction pairs a human agrees is a real disagreement, nothing weaker.
GATE2_LABEL = "contradiction"
CANT_TELL = "cant_tell"
INSUFFICIENT = "insufficient_overlap"
AGREEMENT = "agreement"


@dataclass(frozen=True)
class LabelRow:
    """One committed judgment. Carries no abstract text, so this file is committable."""

    pair_id: str
    derived_label: str
    annotator_label: str
    stratum: str
    reason: str


def join_labels(
    annotations: Mapping[str, Annotation], batch: Sequence[AlamriPair]
) -> list[LabelRow]:
    """Attach each annotator verdict to the pair it was made about.

    REFUSES a mismatch in either direction rather than tolerating it. Both gates are counts
    over what this returns, so a silently dropped row moves a verdict: a missing
    `contradiction` lowers `g` and pushes Gate 2 toward STOP, which is the verdict that
    retires the design. Failing loudly on a bookkeeping error is the only acceptable
    behaviour when the failure mode is "retire a sound design on a parse slip".
    """
    expected = {pair.pair_id for pair in batch}
    missing = sorted(expected - set(annotations))
    if missing:
        raise ValueError(f"annotations missing for {len(missing)} pair(s): {missing}")
    extra = sorted(set(annotations) - expected)
    if extra:
        raise ValueError(f"annotations carry {len(extra)} pair(s) not in the batch: {extra}")

    return [
        LabelRow(
            pair_id=pair.pair_id,
            derived_label=str(pair.label),
            annotator_label=annotations[pair.pair_id].label,
            stratum=pair.stratum,
            reason=annotations[pair.pair_id].reason,
        )
        for pair in batch
    ]


#: Spec §5's numeric pre-commitments, written before any label existed. Expressed as rates so
#: they reproduce the literal pre-committed counts at the design's n (4/5 distractors, 5/10
#: agreement pairs) without pretending to a binomial calibration they do not have -- unlike
#: Gate 2's bands, which ARE such a calibration and therefore refuse any n but 15.
_DISTRACTOR_UNRELATED_RATE = 0.8
_AGREEMENT_STRICTNESS_RATE = 0.5


@dataclass(frozen=True)
class StrictnessRead:
    """The control read. NOT a gate -- it does not decide anything on its own; it says how
    the two pi-hat readings may be reported."""

    insufficient_agreement: int
    n_agreement: int
    insufficient_distractor: int
    n_distractor: int
    agreement_as_agreement: int
    verdict: str


def evaluate_strictness(
    agreement_rows: Sequence[LabelRow], distractor_rows: Sequence[LabelRow]
) -> StrictnessRead:
    """Separate proxy failure from annotator strictness, as far as one annotator allows.

    ADR-0017 recorded that a single annotator cannot distinguish these: Phase 5 applied
    `insufficient_overlap` to 18 of 30 rows, including 4 of 8 gold `agreement` pairs, so the
    pull was general rather than contradiction-specific. The distractors are the addition
    that makes the two readings separable. They are cross-question pairs, genuinely
    unrelated, so `insufficient_overlap` is their correct answer; the agreement controls are
    within-question and genuinely related, so it is not.

    | distractors | agreement controls | verdict |
    |---|---|---|
    | < 80% unrelated | (not reached) | `UNINFORMATIVE` -- the label is not used for
      its meaning at all |
    | >= 80% unrelated | >= 50% insufficient_overlap | `STRICTNESS_CONFOUNDED` |
    | >= 80% unrelated | majority `agreement` | `DISCRIMINATING` |
    | >= 80% unrelated | neither | `UNEXPECTED` |

    `UNEXPECTED` exists so an unanticipated pattern is FLAGGED rather than rounded to the
    nearest named reading. Agreement controls that come back mostly `contradiction` would
    mean something quite different from strictness -- that the derived `agreement` label is
    itself wrong -- and silently folding that into one of the other three verdicts would hide
    the most interesting thing the batch could have said.
    """
    insufficient_agreement = sum(1 for r in agreement_rows if r.annotator_label == INSUFFICIENT)
    as_agreement = sum(1 for r in agreement_rows if r.annotator_label == AGREEMENT)
    insufficient_distractor = sum(1 for r in distractor_rows if r.annotator_label == INSUFFICIENT)
    n_agreement, n_distractor = len(agreement_rows), len(distractor_rows)

    if not n_distractor or insufficient_distractor / n_distractor < _DISTRACTOR_UNRELATED_RATE:
        verdict = "UNINFORMATIVE"
    elif n_agreement and insufficient_agreement / n_agreement >= _AGREEMENT_STRICTNESS_RATE:
        verdict = "STRICTNESS_CONFOUNDED"
    elif n_agreement and as_agreement > n_agreement / 2:
        verdict = "DISCRIMINATING"
    else:
        verdict = "UNEXPECTED"

    return StrictnessRead(
        insufficient_agreement=insufficient_agreement,
        n_agreement=n_agreement,
        insufficient_distractor=insufficient_distractor,
        n_distractor=n_distractor,
        agreement_as_agreement=as_agreement,
        verdict=verdict,
    )


DEFAULT_MARKDOWN = "data/alamri_annotation_batch_1.md"
DEFAULT_BATCH = "evals/gold/alamri_annotation_batch_1_pairs.jsonl"
DEFAULT_LABELS_OUT = "evals/gold/alamri_annotation_batch_1_labels.jsonl"
DEFAULT_LOG = "evals/annotation_runs.jsonl"


def main(argv: list[str] | None = None) -> None:
    """Read the annotated sheet, write the labels, run the gates. No direct unit test by
    convention; `join_labels` and `evaluate_strictness` are tested in their own module and
    both gates were tested in Phase 5's.

    GATE 1 IS A PRECONDITION, NOT A TIEBREAK. On `REVISE_PROTOCOL` this stops without
    computing Gate 2 at all: a high `cant_tell` rate means the protocol was unreadable, and
    nothing can then be concluded about the proxy in either direction. Computing pi-hat
    anyway would produce two numbers that look like readings and are not.
    """
    import argparse
    import json
    from collections import Counter
    from datetime import UTC, datetime
    from pathlib import Path

    from biolit_evals._meta import git_sha
    from biolit_evals.alamri_gold import manifest_hash, read_manifest
    from biolit_evals.annotation_export import evaluate_gate1, evaluate_gate2, parse_annotations

    parser = argparse.ArgumentParser(description="Run the Alamri annotation gates.")
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--labels-out", default=DEFAULT_LABELS_OUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    args = parser.parse_args(argv)

    batch = read_manifest(args.batch)
    annotations = parse_annotations(Path(args.markdown).read_text(encoding="utf-8"))
    rows = join_labels(annotations, batch)

    out = Path(args.labels_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(vars(r), sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"wrote {out}: {len(rows)} annotated pairs")
    print(
        f"annotator composition: {dict(sorted(Counter(r.annotator_label for r in rows).items()))}"
    )

    cant_tell = sum(1 for r in rows if r.annotator_label == CANT_TELL)
    gate1 = evaluate_gate1(cant_tell=cant_tell, n=len(rows))
    lo, hi = gate1.interval
    print(
        f"\nGATE 1 (tractability, all {gate1.n}): cant_tell {gate1.cant_tell}/{gate1.n}, "
        f"95% Wilson [{lo:.4f}, {hi:.4f}] -> {gate1.verdict}"
    )

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "step": "alamri_annotation_gates",
        "batch": Path(args.markdown).name,
        "batch_manifest": args.batch,
        "batch_hash": manifest_hash(batch),
        "n_annotated": len(rows),
        "strata": dict(sorted(Counter(r.stratum for r in rows).items())),
        "derived_composition": dict(sorted(Counter(r.derived_label for r in rows).items())),
        "annotator_composition": dict(sorted(Counter(r.annotator_label for r in rows).items())),
        "gate1": {
            "cant_tell": gate1.cant_tell,
            "n": gate1.n,
            "verdict": gate1.verdict,
            "interval": list(gate1.interval),
        },
    }

    if gate1.verdict == "REVISE_PROTOCOL":
        entry["gate2"] = None
        entry["gate2_skipped_reason"] = (
            "Gate 1 returned REVISE_PROTOCOL. Per the spec's pre-registered rule the Gate 2 "
            "readings are not interpreted at all: an unreadable protocol supports no "
            "conclusion about the proxy in either direction."
        )
        with Path(args.log).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        print(
            "\nGate 1 fired REVISE_PROTOCOL -- Gate 2 NOT computed, per the pre-registered "
            "rule. The protocol needs revision before any pi-hat reading means anything."
        )
        return

    gate2 = {}
    for stratum in (STRATUM_CLEAN, STRATUM_FLAGGED):
        subset = [r for r in rows if r.stratum == stratum]
        genuine = sum(1 for r in subset if r.annotator_label == GATE2_LABEL)
        result = evaluate_gate2(genuine=genuine, n=len(subset))
        gate2[stratum] = result
        lo, hi = result.interval
        print(
            f"GATE 2 ({stratum}): g = {result.genuine}/{result.n}, "
            f"pi-hat = {result.genuine / result.n:.4f}, "
            f"95% Wilson [{lo:.4f}, {hi:.4f}] -> {result.verdict}"
        )

    strictness = evaluate_strictness(
        [r for r in rows if r.stratum == STRATUM_AGREEMENT],
        [r for r in rows if r.stratum == STRATUM_DISTRACTOR],
    )
    print(
        f"\nCONTROLS: agreement pairs insufficient_overlap "
        f"{strictness.insufficient_agreement}/{strictness.n_agreement} "
        f"(as agreement {strictness.agreement_as_agreement}); distractors "
        f"insufficient_overlap {strictness.insufficient_distractor}/{strictness.n_distractor} "
        f"-> {strictness.verdict}"
    )

    entry["gate2"] = {
        stratum: {
            "genuine": r.genuine,
            "n": r.n,
            "pi_hat": r.genuine / r.n,
            "verdict": r.verdict,
            "interval": list(r.interval),
        }
        for stratum, r in gate2.items()
    }
    entry["controls"] = vars(strictness)
    entry["per_stratum_annotator_composition"] = {
        stratum: dict(
            sorted(Counter(r.annotator_label for r in rows if r.stratum == stratum).items())
        )
        for stratum in sorted({r.stratum for r in rows})
    }
    with Path(args.log).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"\nlogged to {args.log}")


if __name__ == "__main__":
    main()
