"""Export the 83 frozen (query, cluster) rows for blind relevance annotation.

Design: `docs/superpowers/specs/2026-09-05-cluster-relevance-annotation-design.md`.

⚠️ CLUSTERS ARE **NOT** MERGED BY KEY HERE, which is the one thing that differs from
`synth_gate_a`. That module merges across the eight runs (Ruling 26) because it draws a single
cross-query sample and one conceptual cluster must not be scored twice. This module's unit is
`(query, cluster)` -- the label answers "is this cluster relevant TO THIS QUESTION" -- so the
query attribution is the measurement, and merging would destroy it. Per-query counts sum to 83.

The annotator sees names, never MeSH ids (§2), and never anything that routes back to the
cluster or to `select_stage`'s decision (§3.2). The reverse mapping lives in a separate manifest
that stays closed until every label is written.
"""

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ExportRow:
    """One `(query, cluster)` judgment. `cluster_key` and `is_distractor` are MANIFEST-ONLY
    and are never emitted by `as_export`."""

    row_id: str
    query: str
    chemical: str
    disease: str
    n_papers: int
    year_range: str
    cluster_key: str
    is_distractor: bool

    def as_export(self) -> dict[str, object]:
        """The annotator-facing record. `label` is None: unlabelled is the starting state,
        and an absent key would let a skipped row look like a deliberate one."""
        return {
            "row_id": self.row_id,
            "query": self.query,
            "chemical": self.chemical,
            "disease": self.disease,
            "n_papers": self.n_papers,
            "year_range": self.year_range,
            "label": None,
        }


def _year_range(low: int | None, high: int | None) -> str:
    if low is None or high is None:
        return "year unknown"
    return str(low) if low == high else f"{low}-{high}"


def build_rows(
    clusters: Sequence[tuple[str, str, Sequence[str], tuple[int | None, int | None]]],
    *,
    names: Mapping[str, str],
    distractors: Sequence[tuple[str, str]] = (),
    rng: random.Random | None = None,
) -> list[ExportRow]:
    """Build annotator rows from `(query, cluster_key, paper_ids, (min_year, max_year))`.

    RAISES on a concept it cannot name. Falling back to the raw id would violate §2's
    no-ids rule on precisely the rows where linking is weakest -- which are the rows most
    worth a human judgment, so degrading them silently is the worst available failure.

    ⚠️ THE SHUFFLE HAPPENS BEFORE `row_id` IS ASSIGNED, and that ordering is the whole point.
    Distractors are appended after the real clusters, so numbering first would put every
    control row in the trailing block of ids -- an annotator reaching the end of the file
    would recognise them on sight and Gate 2 would measure nothing. §3.4 requires them to be
    indistinguishable, which is a property of the ids as much as of the fields.
    """
    rows: list[ExportRow] = []
    marked = [(q, key, pids, years, False) for q, key, pids, years in clusters]
    marked += [(q, key, (), (None, None), True) for q, key in distractors]
    if rng is not None:
        rng.shuffle(marked)
    for index, (query, key, paper_ids, years, is_distractor) in enumerate(marked):
        chemical_id, disease_id = key.split("|")
        for concept_id in (chemical_id, disease_id):
            if concept_id not in names:
                raise ValueError(
                    f"build_rows: no name for concept {concept_id!r} in cluster {key!r}. "
                    "Refusing to fall back to the raw id: the annotator must never see one "
                    "(design §2), and a row that cannot be named is a linking defect worth "
                    "recording rather than papering over."
                )
        rows.append(
            ExportRow(
                row_id=f"r{index:03d}",
                query=query,
                chemical=names[chemical_id],
                disease=names[disease_id],
                n_papers=len(paper_ids),
                year_range=_year_range(*years),
                cluster_key=key,
                is_distractor=is_distractor,
            )
        )
    return rows


def choose_distractors(
    per_query: Mapping[str, Sequence[str]], *, n: int = 8, seed: int = 20260905
) -> list[tuple[str, str]]:
    """Pick `n` (query, cluster_key) pairs where the cluster belongs to a DIFFERENT query.

    ADR-0018's control, and it only works if the pairing is genuinely unrelated: a cluster
    shown under its own query is an ordinary row, and labelling it `answers` would read as
    annotator failure when it was in fact correct. Sorted before shuffling so the draw is
    reproducible from the seed regardless of dict ordering.
    """
    rng = random.Random(seed)
    queries = sorted(per_query)
    pairs = sorted(
        (shown_under, key)
        for shown_under in queries
        for source in queries
        if source != shown_under
        for key in per_query[source]
        if key not in per_query[shown_under]
    )
    if len(pairs) < n:
        raise ValueError(
            f"choose_distractors: only {len(pairs)} cross-query pairs available, need {n}."
        )
    rng.shuffle(pairs)
    # SPREAD ACROSS QUERIES: at most one distractor per question on the first pass. A control
    # concentrated in one query would test discrimination only in that query's subject area,
    # and Gate 2 reads as a single 8-row verdict about the annotator generally.
    picked: list[tuple[str, str]] = []
    used_under: set[str] = set()
    for shown_under, key in pairs:
        if shown_under in used_under:
            continue
        picked.append((shown_under, key))
        used_under.add(shown_under)
        if len(picked) == n:
            return picked
    # Only reached when n exceeds the number of queries; top up in the same shuffled order.
    for pair in pairs:
        if pair not in picked:
            picked.append(pair)
            if len(picked) == n:
                break
    return picked


def rows_hash(rows: Sequence[ExportRow]) -> str:
    """Content-addressed over the rows, stable across order and serialization.

    Same posture as `synth_corpus.sample_hash` and `contradiction_gold.manifest_hash`: this
    pins WHICH rows were put in front of the annotator, so a later reading cannot be
    quietly computed over a different row set than the one that was frozen.
    """
    payload = json.dumps(
        sorted((r.query, r.cluster_key, r.n_papers, r.year_range, r.is_distractor) for r in rows),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


#: Design §2. `cant_tell` is Gate 1's instrument, not a convenience escape hatch.
RELEVANCE_LABELS = frozenset({"answers", "background", "off_topic", "cant_tell"})


def render_markdown(rows: Sequence[ExportRow], *, rows_hash: str, seed: int) -> str:
    """The annotator-facing sheet, in the exact block format `parse_annotations` reads.

    Numbered `##` heading carrying the backticked `row_id`, then a fenced `label:`/`reason:`
    stub -- matching `annotation_export._PAIR_HEADING` so that parser is reused UNCHANGED.
    Reusing it is the point: it refuses a missing or unrecognised label rather than skipping
    the block, and a silently dropped row moves a gate count directly.

    ⚠️ EVERY ROW HAS THE SAME SHAPE, controls included, and the preamble names neither the
    number of controls nor the strata. A reader who knew there were exactly eight could work
    backwards from the counts; the hash and seed are enough to reproduce the row set after
    labelling, which is when reproducibility is needed.
    """
    lines = [
        "# Cluster relevance — blind annotation",
        "",
        f"Rows hash {rows_hash}, seed {seed}, {len(rows)} rows.",
        "",
        "Each row shows a **question** and one **cluster** — a chemical concept and a "
        "disease concept that some set of papers discussed in the same sentence. Judge "
        "whether that cluster is relevant to that question, and record one of:",
        "",
        "- `answers` — this cluster is part of what the question asked for.",
        "- `background` — genuinely about the question's subject, but not what was asked: "
        "the drug's indication, its main comparator, a co-occurring condition.",
        "- `off_topic` — not what the question was about.",
        "- `cant_tell` — the question, the concepts, or the pairing is too ambiguous to judge.",
        "",
        "Add a one-line reason. Judge each row on its own; do not go back and revise earlier "
        "rows once later ones clarify the distinctions, because that turns a blind pass into "
        "a calibrated one.",
        "",
        "Nothing on this sheet encodes the answer: every row has the same shape, whatever "
        "the cluster is.",
        "",
        "---",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines += [
            f"## {index}. `{row.row_id}`",
            "",
            f"**Question:** {row.query}",
            "",
            f"**Cluster:** {row.chemical} — {row.disease}",
            "",
            f"{row.n_papers} papers, {row.year_range}.",
            "",
            "```",
            "label:",
            "reason:",
            "```",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


DEFAULT_STATES = "data/synth/states"
DEFAULT_OUT = "data/relevance"


def main(argv: list[str] | None = None) -> None:
    """Freeze the annotation rows. No network, no model, no ranker.

    `main()` gets no direct unit test per project convention; every function it calls is
    tested in its own module.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Export blind cluster-relevance rows.")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required: it fixes row order and the distractor draw. Never defaulted.",
    )
    args = parser.parse_args(argv)

    # NOT merged by key. The unit is (query, cluster); merging across runs is what
    # `synth_gate_a` does for its cross-query sample, and it would destroy the attribution
    # this measurement is about.
    clusters: list[tuple[str, str, list[str], tuple[int | None, int | None]]] = []
    per_query: dict[str, list[str]] = {}
    names: dict[str, str] = {}
    for path in sorted(Path(args.states).glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        question = state["question"]
        years = {p["id"]: p.get("year") for p in state["candidate_papers"]}
        for record in state["extracted_records"].values():
            for entity in record["entities"]:
                if entity.get("canonical_id") and entity.get("canonical_name"):
                    names.setdefault(entity["canonical_id"], entity["canonical_name"])
        for cluster in state["clusters"]:
            known = [years.get(pid) for pid in cluster["paper_ids"]]
            present = [y for y in known if y is not None]
            span = (min(present), max(present)) if present else (None, None)
            clusters.append((question, cluster["key"], cluster["paper_ids"], span))
            per_query.setdefault(question, []).append(cluster["key"])

    distractors = choose_distractors(per_query, n=args.distractors, seed=args.seed)
    rows = build_rows(clusters, names=names, distractors=distractors, rng=random.Random(args.seed))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path, manifest_path = out / "rows.jsonl", out / "manifest.json"
    sheet_path = out / "sheet.md"
    sheet_path.write_text(
        render_markdown(rows, rows_hash=rows_hash(rows), seed=args.seed), encoding="utf-8"
    )
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.as_export()) + "\n")
    manifest_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "rows_hash": rows_hash(rows),
                "n_real": len(clusters),
                "n_distractors": len(distractors),
                "rows": {
                    r.row_id: {
                        "query": r.query,
                        "cluster_key": r.cluster_key,
                        "is_distractor": r.is_distractor,
                    }
                    for r in rows
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{len(clusters)} real rows + {len(distractors)} distractors = {len(rows)}")
    print(f"rows_hash {rows_hash(rows)}  seed {args.seed}")
    print(f"wrote {sheet_path}   <-- ANNOTATE HERE")
    print(f"wrote {rows_path}")
    print(f"wrote {manifest_path}  <-- DO NOT OPEN until every label is written")


if __name__ == "__main__":
    main()
