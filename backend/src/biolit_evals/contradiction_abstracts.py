"""Step 2 of the spec's build order: fetch abstracts for the pool, then apply the
pre-committed drop rule.

Abstracts are fetched, never committed -- same posture as BC5CDR, and it avoids committing
copyrighted text. The cache lives under gitignored `data/`.

This module is a `main()` driver only; per this project's convention `main()` gets no direct
unit test. Every piece of logic it depends on is tested elsewhere: `available_abstracts`,
`compose_corpus` and `drop_report` in `test_contradiction_corpus.py`, and
`PubMedClient.efetch_abstracts` in `test_pubmed.py`.
"""

DEFAULT_POOL = "evals/gold/contradiction_pairs.jsonl"
DEFAULT_CORPUS = "evals/gold/contradiction_corpus.jsonl"
DEFAULT_CACHE = "data/contradiction_abstracts.json"
DEFAULT_LOG = "evals/contradiction_corpus_runs.jsonl"

# NCBI allows 3 requests/second without an API key. efetch takes many ids per call, so the
# whole 5,400-paper corpus is ~27 requests; the sleep keeps us under the limit anyway.
BATCH_SIZE = 200
SLEEP_BETWEEN_BATCHES_S = 0.4


def main(argv: list[str] | None = None) -> None:
    # Heavy imports local to main, same pattern as cluster_eval.main.
    import argparse
    import asyncio
    import json
    import statistics
    from datetime import UTC, datetime
    from pathlib import Path

    import httpx

    from biolit.clients.pubmed import PubMedClient
    from biolit.config import get_settings
    from biolit_evals._meta import git_sha
    from biolit_evals.contradiction_corpus import (
        available_abstracts,
        compose_corpus,
        drop_report,
    )
    from biolit_evals.contradiction_gold import manifest_hash, read_manifest, write_manifest

    parser = argparse.ArgumentParser(description="Fetch abstracts and apply the drop rule.")
    parser.add_argument("--pool", default=DEFAULT_POOL)
    parser.add_argument("--out", default=DEFAULT_CORPUS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="Target N per class. The pool is 3x this, per the spec's topping-up rule.",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=200,
        help="Below this a class triggers topping up (spec's rule table, argued from "
        "interval width at p=0.5).",
    )
    args = parser.parse_args(argv)

    pool = read_manifest(args.pool)
    pmids = sorted({p for pair in pool for p in (pair.paper_id_a, pair.paper_id_b)})

    # The cache holds the fetch result verbatim, INCLUDING papers with no abstract. Caching
    # only the successes would make every re-run retry the permanent failures, and would hide
    # how many there were.
    cache_path = Path(args.cache)
    fetched: dict[str, tuple[str | None, int | None]] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        fetched = {pmid: (rec["abstract"], rec["year"]) for pmid, rec in raw.items()}
    missing = [p for p in pmids if p not in fetched]
    print(f"{len(pmids)} pmids in pool; {len(fetched)} cached; fetching {len(missing)}")

    async def fetch_all() -> None:
        async with httpx.AsyncClient(timeout=120) as http:
            client = PubMedClient(http, get_settings())
            for start in range(0, len(missing), BATCH_SIZE):
                batch = missing[start : start + BATCH_SIZE]
                got = await client.efetch_abstracts(batch)
                # A pmid PubMed does not return at all is recorded as an explicit miss, so it
                # is not retried forever and is counted honestly in the drop rate.
                for pmid in batch:
                    fetched[pmid] = got.get(pmid, (None, None))
                print(f"  fetched {min(start + BATCH_SIZE, len(missing))}/{len(missing)}")
                await asyncio.sleep(SLEEP_BETWEEN_BATCHES_S)

    if missing:
        asyncio.run(fetch_all())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {p: {"abstract": a, "year": y} for p, (a, y) in sorted(fetched.items())},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    abstracts = available_abstracts(fetched)
    years = {p: y for p, (_a, y) in fetched.items() if y is not None}
    report = drop_report(pool, abstracts, years)
    composed = compose_corpus(pool, abstracts, per_class=args.per_class, floor=args.floor)
    write_manifest(composed.pairs, args.out)

    def _summary(values: list[int]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "n": len(values),
            "median": statistics.median(values),
            "mean": round(statistics.fmean(values), 1),
            "min": min(values),
            "max": max(values),
        }

    line = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "step": "abstract_fetch",
        "pool_manifest": str(args.pool),
        "corpus_manifest": str(args.out),
        "corpus_hash": manifest_hash(composed.pairs),
        "per_class_target": args.per_class,
        "floor": args.floor,
        "n_pmids": len(pmids),
        "n_pmids_with_abstract": len(abstracts),
        "kept_of_total_by_class": {k: list(v) for k, v in sorted(report.per_class.items())},
        "n_by_class": dict(sorted(composed.n_by_class.items())),
        "outcome_by_class": {k: str(v) for k, v in sorted(composed.outcome_by_class.items())},
        # Reported REGARDLESS of outcome: if availability correlates with era or indexing
        # quality, filtering one class harder makes its papers systematically unlike the
        # others' for reasons unrelated to the label. Limitation 5.
        "abstract_length_by_class": {
            k: _summary(v) for k, v in sorted(report.length_by_class.items())
        },
        "year_by_class": {k: _summary(v) for k, v in sorted(report.year_by_class.items())},
    }
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    print(json.dumps(line, indent=2))


if __name__ == "__main__":
    main()
