"""The blind annotation sheet and the two gates for the contradiction-detection eval.

The gold standard labels a pair `contradiction` when CTD records opposite evidence
directions for its two papers. That is a PROXY. Pi -- the fraction of such pairs a human
annotator agrees is a genuine disagreement -- is unknown until measured, and it can only be
measured if the annotator cannot see the gold label or the CTD directions while judging: a
leaked label makes pi unmeasurable, and nothing downstream could detect that it had
happened. `export_blind_sheet` is the whole protocol for keeping that from occurring, per
the ADR-0006 `domain_sample` blind-annotation precedent.

Two gates read the first annotated batch (30 pairs: 15 contradiction + 15 other) and decide
whether continued spending on the proxy is justified. `evaluate_gate1` is a TRACTABILITY
check over the whole batch -- can the protocol be judged at all, or is `cant_tell` swallowing
it. `evaluate_gate2` is a STOP RULE over the 15 contradiction pairs only -- is the proxy
badly wrong. The two gates read different subsets of the same batch and have different `n`
by design; see each function's docstring.
"""

import random
import re
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
    concept_names: Mapping[str, str] | None = None,
) -> list[dict]:
    """Build the annotator-facing sheet: `n_contradiction` contradiction pairs plus
    `n_other` pairs spread as evenly as possible across whichever non-contradiction labels
    are present in `pairs` (agreement, insufficient_overlap), with the remainder -- if
    `n_other` does not divide evenly -- going to the alphabetically-earlier label.

    Selection within each label takes pairs in the order they appear in `pairs`, so it is
    reproducible from that order alone; the ONE random step is the final shuffle, which
    exists purely so an annotator cannot infer a row's class from its position on the sheet.
    This "first N" selection is unbiased only if `pairs` is ALREADY in random order --
    true when passed the manifest (`sample_pairs` shuffles it before writing), but a caller
    passing raw candidates sorted by endpoint (e.g. `build_candidates`'s output directly)
    would get a systematically biased sample instead of a random one.

    Each row carries exactly `{"pair_id", "shared_concept_id", "shared_concept_name",
    "abstract_a", "abstract_b"}` -- no `label`, no `direction_a`/`direction_b`, and crucially
    exactly ONE concept on every row regardless of class. `pair_id` is
    `f"{paper_id_a}_{paper_id_b}"`, which is unambiguous only because PMIDs are purely numeric
    and never contain `_`.

    ONE CONCEPT, NOT THE KEY, AND THE REASON IS THE WHOLE POINT. Emitting `chemical_id` and
    `disease_id` leaks the gold answer for a third of the corpus: `insufficient_overlap` is
    DEFINED as sharing a single endpoint rather than a curated key, so its pairs are the only
    ones carrying a null slot. Measured on the real 900-pair corpus, 300/300
    insufficient_overlap pairs have a null endpoint against 0/600 of the other two classes --
    a null slot identified the class with certainty. An annotator who can rule that class out
    on sight chooses between two labels instead of three, which can only push pi UP, and pi is
    the ceiling every other number in this eval is compared against.

    WHICH endpoint is shown, when both exist, is chosen with `rng` rather than fixed. That is
    not arbitrary: MeSH ids do not reveal whether they denote a chemical or a disease (in this
    corpus chemicals are 641 `D`- and 170 `C`-prefixed, diseases 687 `D`- and 2 `C`-prefixed),
    but `shared_concept_name` plainly does -- a reader knows "Vitamin D" from "Nausea". Always
    showing the chemical for two-endpoint pairs would therefore make any disease-NAMED concept
    a fresh tell for insufficient_overlap, which shares its disease in 89 of 300 cases.
    Choosing at random leaves no row classifiable with certainty.

    `concept_names` is optional; a missing name renders empty rather than omitting the field,
    so every row keeps an identical key set.
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

    names = concept_names or {}
    rows: list[dict] = []
    for pair in selected:
        endpoints = [e for e in (pair.chemical_id, pair.disease_id) if e]
        concept = endpoints[0] if len(endpoints) == 1 else rng.choice(endpoints)
        rows.append(
            {
                "pair_id": f"{pair.paper_id_a}_{pair.paper_id_b}",
                "shared_concept_id": concept,
                "shared_concept_name": names.get(concept, ""),
                "abstract_a": abstracts[pair.paper_id_a],
                "abstract_b": abstracts[pair.paper_id_b],
            }
        )
    return rows


@dataclass(frozen=True)
class Gate2:
    """The result of `evaluate_gate2`. `interval` is a 95% Wilson interval over `genuine`
    of `n` -- reported so a reader can SEE how wide it is, not so the gate can act on it:
    see `evaluate_gate2` for why this is a stop rule and not a measurement of pi."""

    genuine: int
    n: int
    verdict: str
    interval: tuple[float, float]


_GATE2_N = 15


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

    REFUSES n != 15. The bands are literal counts derived from binomial tail probabilities
    computed at exactly n=15, not a proportional rule -- unlike Gate 1's rate threshold (see
    `evaluate_gate1`), they do not transfer to another n. This matters in practice because
    the design's human-facing unit is "the first 30-pair batch", so 30 -- the whole batch,
    not the 15-pair contradiction-only subset this gate is calibrated on -- is the number
    most likely to be passed by mistake. A `Gate2` silently built for the wrong n would carry
    exactly as much apparent authority as a correct one, so this raises instead. `genuine` is
    likewise bounded to `[0, n]`: a mis-transcribed count is a smaller version of the same
    silent-wrong-answer risk, and the check is cheap.
    """
    if n != _GATE2_N:
        raise ValueError(
            f"evaluate_gate2: n={n} is not the calibrated batch size (n=15). The verdict "
            "bands are literal counts derived from binomial tail probabilities computed at "
            "n=15, not a proportional rule -- they do not transfer to another n, including "
            "30 (the whole first batch, which is Gate 1's domain; see evaluate_gate1)."
        )
    if not 0 <= genuine <= n:
        raise ValueError(f"evaluate_gate2: genuine={genuine} must be within [0, n={n}].")
    if genuine <= 7:
        verdict = "STOP"
    elif 8 <= genuine <= 10:
        verdict = "CONTINUE_FLAGGED"
    else:
        verdict = "CONTINUE"
    return Gate2(genuine=genuine, n=n, verdict=verdict, interval=wilson_interval(genuine, n))


@dataclass(frozen=True)
class Gate1:
    """The result of `evaluate_gate1`. Mirrors `Gate2`'s shape (a count, `n`, a `verdict`,
    and a `wilson_interval` on the rate) but is a DIFFERENT gate over a DIFFERENT population:
    Gate 1 runs over the whole first batch -- all 30 pairs, across all three classes -- while
    Gate 2 runs over the 15 contradiction pairs within it only. The two gates have different
    `n` by design; they are not interchangeable and neither `n` is a typo for the other."""

    cant_tell: int
    n: int
    verdict: str
    interval: tuple[float, float]


def evaluate_gate1(cant_tell: int, n: int) -> Gate1:
    """Gate 1 is a TRACTABILITY check: does the blind protocol work at all -- do the
    abstracts carry enough information for an annotator to judge, or is `cant_tell` quietly
    swallowing the batch? It runs over the whole first batch (`n` = 30: all three classes),
    not the 15 contradiction-only pairs Gate 2 looks at.

    A `cant_tell` rate above ~1/3 means the protocol needs revision before more annotation
    time is spent: `verdict` is `"REVISE_PROTOCOL"` when `cant_tell / n > 1/3`, else
    `"TRACTABLE"`. Computed as `3 * cant_tell > n` (cross-multiplied) rather than a float
    division, so the boundary is exact: at n=30, cant_tell=10 is precisely 1/3 and passes.

    UNLIKE Gate 2, this function accepts a generic `n` rather than refusing anything but 15.
    That is a deliberate difference, not an oversight: Gate 2's bands are literal counts
    calibrated by a binomial-tail calculation done at exactly n=15, so they do not transfer
    to another n. Gate 1's 1/3 threshold is a PROPORTIONAL rate, which by construction means
    the same thing at any n -- so refusing an "uncalibrated" n here would be refusing
    something that was never miscalibrated in the first place.
    """
    verdict = "REVISE_PROTOCOL" if 3 * cant_tell > n else "TRACTABLE"
    return Gate1(cant_tell=cant_tell, n=n, verdict=verdict, interval=wilson_interval(cant_tell, n))


#: The four verdicts an annotator may write. `cant_tell` is deliberately NOT a
#: `ContradictionLabel` -- it is an annotator-only escape hatch with no gold counterpart, and
#: counting it is the entire job of Gate 1.
ANNOTATOR_LABELS = frozenset(
    {str(label) for label in ContradictionLabel} | {"cant_tell"},
)

_PAIR_HEADING = re.compile(r"^## \d+\.\s+`([^`]+)`\s*$", re.MULTILINE)
_LABEL_LINE = re.compile(r"^\s*label:\s*(\S+)\s*$", re.MULTILINE)
_REASON_LINE = re.compile(r"^\s*reason:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Annotation:
    label: str
    reason: str


def parse_annotations(sheet: str) -> dict[str, Annotation]:
    """Read the annotator's verdicts back out of the markdown sheet `main` wrote.

    The inverse of `export_blind_sheet`. Both gates are COUNTS over what this returns, so a
    block this function silently skips or mis-reads moves a gate verdict directly. It
    therefore refuses rather than tolerating: a block with no `label:` line raises, and so
    does a label outside `ANNOTATOR_LABELS`.

    That strictness is asymmetric on purpose. A tolerant parser drops what it cannot read,
    and a dropped `contradiction` lowers `g` -- pushing Gate 2 toward STOP, the verdict that
    retires the design. The failure mode of a lenient parser here is "retire a sound design
    on a typo", which is the one outcome that must not be reachable quietly.
    """
    headings = list(_PAIR_HEADING.finditer(sheet))
    out: dict[str, Annotation] = {}
    for index, heading in enumerate(headings):
        pair_id = heading.group(1)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(sheet)
        body = sheet[heading.end() : end]

        label_match = _LABEL_LINE.search(body)
        if label_match is None:
            raise ValueError(f"pair {pair_id} carries no `label:` line")
        label = label_match.group(1)
        if label not in ANNOTATOR_LABELS:
            raise ValueError(
                f"pair {pair_id} has label {label!r}, which is not one of "
                f"{sorted(ANNOTATOR_LABELS)}"
            )

        reason_match = _REASON_LINE.search(body)
        out[pair_id] = Annotation(label=label, reason=reason_match.group(1) if reason_match else "")
    return out


DEFAULT_CORPUS = "evals/gold/contradiction_corpus.jsonl"
DEFAULT_TEXTS = "data/contradiction_abstract_texts.json"
DEFAULT_ALIASES = "data/canon/mesh_aliases.json.gz"
# The sheet carries full abstract text, so it is written under gitignored data/ and never
# committed -- the same posture the fetch step takes. Only the ANNOTATOR'S LABELS, which
# contain no abstract text, become a committed artifact.
DEFAULT_SHEET = "data/annotation_batch_1.json"
DEFAULT_MARKDOWN = "data/annotation_batch_1.md"


def _concept_names(aliases_path: str) -> dict[str, str]:
    """{normalized MeSH id: canonical name} inverted from the Phase 3A alias artifact.

    The artifact is alias-major -- {alias: [[MESH:id, canonical_name, is_primary], ...]} --
    and its ids carry the `MESH:` prefix the corpus strips, so both are normalised here.
    Primary aliases win; a non-primary one is used only if nothing else names the concept.
    """
    import gzip
    import json

    names: dict[str, str] = {}
    with gzip.open(aliases_path, "rt", encoding="utf-8") as fh:
        for entries in json.load(fh).values():
            for raw_id, canonical, is_primary in entries:
                key = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
                if is_primary or key not in names:
                    names[key] = canonical
    return names


def main(argv: list[str] | None = None) -> None:
    """Write the first blind annotation batch. `main()` gets no direct unit test by
    convention; `export_blind_sheet` and both gates are tested in their own module."""
    import argparse
    import json
    from pathlib import Path

    from biolit_evals.contradiction_gold import manifest_hash, read_manifest

    parser = argparse.ArgumentParser(description="Export the blind annotation batch.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--texts", default=DEFAULT_TEXTS)
    parser.add_argument("--aliases", default=DEFAULT_ALIASES)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN)
    parser.add_argument("--n-contradiction", type=int, default=15)
    parser.add_argument("--n-other", type=int, default=15)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Shuffle seed. Required: it fixes which pairs are on the sheet and in what order.",
    )
    args = parser.parse_args(argv)

    pairs = read_manifest(args.corpus)
    texts = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    rows = export_blind_sheet(
        pairs,
        texts,
        n_contradiction=args.n_contradiction,
        n_other=args.n_other,
        rng=random.Random(args.seed),
        concept_names=_concept_names(args.aliases),
    )

    sheet = Path(args.sheet)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Blind annotation batch 1",
        "",
        f"Corpus `{args.corpus}` (hash {manifest_hash(pairs)}), seed {args.seed}, "
        f"{len(rows)} pairs.",
        "",
        "For each pair, judge **from the two abstracts alone** whether they disagree about "
        "the shared concept, then record one of `contradiction`, `agreement`, "
        "`insufficient_overlap`, or `cant_tell`, with a one-line reason.",
        "",
        "Nothing on this sheet encodes the gold answer: every row shows exactly one shared "
        "concept, and which endpoint that is varies by design.",
        "",
        "---",
        "",
    ]
    for i, row in enumerate(rows, start=1):
        name = row["shared_concept_name"] or "(unnamed)"
        lines += [
            f"## {i}. `{row['pair_id']}`",
            "",
            f"**Shared concept:** {name} (`{row['shared_concept_id']}`)",
            "",
            "**Abstract A**",
            "",
            row["abstract_a"],
            "",
            "**Abstract B**",
            "",
            row["abstract_b"],
            "",
            "```",
            "label:",
            "reason:",
            "```",
            "",
            "---",
            "",
        ]
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {sheet} and {args.markdown}: {len(rows)} pairs")
    named = sum(1 for r in rows if r["shared_concept_name"])
    print(f"rows with a resolved concept name: {named}/{len(rows)}")


if __name__ == "__main__":
    main()
