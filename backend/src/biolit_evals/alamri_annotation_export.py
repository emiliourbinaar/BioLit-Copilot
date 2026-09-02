"""The blind annotation sheet for the Alamri pi-hat pass.

Phase 5's `export_blind_sheet` cannot be reused: it is CTD-specific, taking `GoldPair` with
`chemical_id`/`disease_id` and solving a leak (the null-endpoint tell) that does not exist
here. What IS reused, unmodified, is everything downstream -- `parse_annotations`,
`evaluate_gate1`, `evaluate_gate2`, `wilson_interval` -- so the markdown this module writes
is deliberately in the exact block format `parse_annotations` already reads.

The protocol differs from Phase 5's in two ways, each fixing a defect ADR-0017 recorded.
The sheet shows the review's CLINICAL QUESTION, which names population, intervention,
comparator and outcome -- the context the single random MeSH endpoint failed to convey. And
it shows FULL ABSTRACTS rather than the annotator-extracted claim sentence: the Critic only
ever sees abstracts, so scoring anything else measures a judgment the arm never makes, and
the claim sentence is what Alamri's own annotators keyed on, so displaying it would hand
over the original judgment instead of eliciting an independent one.
"""

import random
from collections.abc import Mapping, Sequence

from biolit_evals.alamri_gold import AlamriPair

#: The keys every row carries, on every stratum. Asserted as a SET in the tests rather than
#: checked field by field: a key present on some strata and absent on others is a structural
#: tell even when its value is innocuous.
ROW_KEYS = ("pair_id", "question", "abstract_a", "abstract_b")


def export_stratified_sheet(
    batch: Sequence[AlamriPair],
    abstracts: Mapping[str, str],
    questions: Mapping[str, str],
    *,
    rng: random.Random,
) -> list[dict]:
    """Render the sampled batch as annotator-facing rows, shuffled and de-tell'd.

    EXACTLY ONE QUESTION PER ROW, INCLUDING DISTRACTORS. A distractor joins two papers from
    different reviews, so the obvious rendering shows both questions -- and that is a
    structural tell: a two-question row is a distractor with certainty, and this project's
    sole annotator knows five of them are in the batch. Showing one question turns the
    distractor into an honest instance of the same task every other row poses ("do these two
    abstracts disagree about this question?"), whose correct answer happens to be
    `insufficient_overlap` because the second paper does not address the question at all.
    Which of the two questions is shown is chosen with `rng`, so the displayed question is
    not systematically the first paper's.

    THE TWO ABSTRACTS ARE PRESENTED IN RANDOM ORDER. In the manifest, `paper_id_a` of a
    contradiction pair is always the `YS` paper and `paper_id_b` always the `NO` paper. Left
    unshuffled, "abstract A answers yes" would hold across every contradiction row in the
    batch -- a regularity a careful annotator could pick up over 45 rows, and one that says
    nothing about whether the two papers disagree. The swap is display-only: `pair_id` still
    identifies the pair, so `parse_annotations` maps verdicts back regardless.
    """
    rows: list[dict] = []
    for pair in batch:
        question_key = rng.choice((pair.question_id_a, pair.question_id_b))
        first, second = pair.paper_id_a, pair.paper_id_b
        if rng.random() < 0.5:
            first, second = second, first
        rows.append(
            {
                "pair_id": pair.pair_id,
                "question": questions[question_key],
                "abstract_a": abstracts[first],
                "abstract_b": abstracts[second],
            }
        )
    rng.shuffle(rows)
    return rows


def render_markdown(rows: Sequence[Mapping[str, str]], *, corpus_hash: str, seed: int) -> str:
    """The annotator-facing sheet, in the exact block format `parse_annotations` reads.

    The heading is a numbered `##` carrying the backticked `pair_id`, and each block ends
    with a fenced `label:` / `reason:` stub, because that is what the Phase 5 parser's
    regexes match -- see `annotation_export._PAIR_HEADING`. Matching an
    existing parser is the point: reusing it unmodified is what keeps the gates the same
    instrument that produced ADR-0017's reading.

    The preamble names neither the strata nor their sizes. A reader who knew the batch held
    exactly five distractors and ten agreement controls could work backwards from the
    counts; the seed and corpus hash are enough to reproduce the sample afterwards.
    """
    lines = [
        "# Alamri blind annotation batch 1",
        "",
        f"Corpus hash {corpus_hash}, seed {seed}, {len(rows)} pairs.",
        "",
        "For each pair, judge **from the two abstracts alone** whether the two papers "
        "disagree about the clinical question shown, then record one of `contradiction`, "
        "`agreement`, `insufficient_overlap`, or `cant_tell`, with a one-line reason.",
        "",
        "Use `insufficient_overlap` when the two abstracts do not address a common "
        "proposition closely enough to agree or disagree, and `cant_tell` when they might "
        "but the abstracts do not say enough to judge.",
        "",
        "Nothing on this sheet encodes the answer: every row shows one clinical question "
        "and two abstracts, in the same shape, whatever the pair is.",
        "",
        "---",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines += [
            f"## {index}. `{row['pair_id']}`",
            "",
            f"**Clinical question:** {row['question']}",
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
    return "\n".join(lines)


DEFAULT_CORPUS = "data/alamri/corpus.xml"
DEFAULT_ABSTRACT_CACHE = "data/alamri/abstracts.json"
DEFAULT_CANDIDATES = "evals/gold/alamri_contradiction_pairs.jsonl"
DEFAULT_BATCH = "evals/gold/alamri_annotation_batch_1_pairs.jsonl"
# The sheet carries full abstract text, so it lives under gitignored data/ and is never
# committed. Only ids, strata and (later) the annotator's labels become committed artifacts.
DEFAULT_SHEET = "data/alamri_annotation_batch_1.json"
DEFAULT_MARKDOWN = "data/alamri_annotation_batch_1.md"

_FETCH_BATCH_SIZE = 100
_SLEEP_BETWEEN_BATCHES_S = 0.5


def main(argv: list[str] | None = None) -> None:
    """Freeze the derived pairs and write the blind batch. No direct unit test by
    convention; every function it calls is tested in its own module.

    Free: one NCBI efetch pass over at most 254 pmids, cached. No LLM call, no credential,
    no authorization gate -- there is nothing to authorize.
    """
    import argparse
    import asyncio
    import json
    from collections import Counter
    from pathlib import Path

    import httpx

    from biolit.clients.pubmed import PubMedClient
    from biolit.config import get_settings
    from biolit_evals.alamri_gold import (
        DEFAULT_QUOTAS,
        build_distractors,
        build_pairs,
        manifest_hash,
        parse_corpus,
        question_id,
        sample_batch,
        write_manifest,
    )

    parser = argparse.ArgumentParser(description="Export the Alamri blind annotation batch.")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--abstracts", default=DEFAULT_ABSTRACT_CACHE)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument("--markdown", default=DEFAULT_MARKDOWN)
    parser.add_argument("--cap-paper", type=int, default=1)
    parser.add_argument("--cap-question", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required: it fixes which pairs are on the sheet, in what order, which of a "
        "distractor's two questions is shown, and which abstract is presented first.",
    )
    args = parser.parse_args(argv)

    claims = parse_corpus(Path(args.corpus).read_text(encoding="utf-8"))
    questions = {question_id(c.question): c.question for c in claims}
    pmids = sorted({c.pmid for c in claims})

    cache_path = Path(args.abstracts)
    fetched: dict[str, str | None] = {}
    if cache_path.exists():
        fetched = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [p for p in pmids if p not in fetched]
    print(f"{len(pmids)} pmids in corpus; {len(fetched)} cached; fetching {len(missing)}")

    async def fetch_all() -> None:
        async with httpx.AsyncClient(timeout=120) as http:
            client = PubMedClient(http, get_settings())
            for start in range(0, len(missing), _FETCH_BATCH_SIZE):
                chunk = missing[start : start + _FETCH_BATCH_SIZE]
                got = await client.efetch_abstracts(chunk)
                # A pmid PubMed does not return is recorded as an explicit miss, so it is not
                # retried forever and is counted honestly rather than silently dropped.
                for pmid in chunk:
                    fetched[pmid] = got.get(pmid, (None, None))[0]
                await asyncio.sleep(_SLEEP_BETWEEN_BATCHES_S)

    if missing:
        asyncio.run(fetch_all())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(dict(sorted(fetched.items())), ensure_ascii=False), encoding="utf-8"
        )

    abstracts = {pmid: text for pmid, text in fetched.items() if text}
    print(f"abstracts available: {len(abstracts)}/{len(pmids)}")

    candidates = build_pairs(claims)
    write_manifest(candidates, args.candidates)
    print(
        f"wrote {args.candidates}: {len(candidates)} derived pairs, hash "
        f"{manifest_hash(candidates)}"
    )

    # Sampling draws only from pairs whose BOTH abstracts are in hand. Filtering after the
    # draw would silently shrink a stratum below the n Gate 2 is calibrated on.
    renderable = [
        p
        for p in candidates + build_distractors(claims)
        if p.paper_id_a in abstracts and p.paper_id_b in abstracts
    ]
    batch = sample_batch(
        renderable,
        rng=random.Random(args.seed),
        quotas=DEFAULT_QUOTAS,
        cap_paper=args.cap_paper,
        cap_question=args.cap_question,
    )
    write_manifest(batch, args.batch)
    batch_hash = manifest_hash(batch)

    rows = export_stratified_sheet(batch, abstracts, questions, rng=random.Random(args.seed))
    sheet_path = Path(args.sheet)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(
        render_markdown(rows, corpus_hash=batch_hash, seed=args.seed), encoding="utf-8"
    )

    print(
        f"wrote {args.batch}, {sheet_path} and {args.markdown}: {len(rows)} pairs, "
        f"batch hash {batch_hash}"
    )
    print(f"strata: {dict(sorted(Counter(p.stratum for p in batch).items()))}")
    appearances = Counter(pid for p in batch for pid in (p.paper_id_a, p.paper_id_b))
    print(f"distinct papers {len(appearances)}, max appearances {max(appearances.values())}")


if __name__ == "__main__":
    main()
