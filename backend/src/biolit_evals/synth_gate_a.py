"""Run the Gate A template arm over the frozen cluster sample.

TEMPLATE ARM ONLY. The LLM arm is deliberately absent: the spec's §7 makes the pilot and the
full run separate authorisation steps, and this module must be able to run to completion
with no credential present and no call made.
"""

DEFAULT_STATES = "data/synth/states"
DEFAULT_ALIASES = "data/canon/mesh_aliases.json.gz"
DEFAULT_OUT = "data/synth/gate_a_template.json"
DEFAULT_LOG = "evals/synth_runs.jsonl"


def main(argv: list[str] | None = None) -> None:
    """Score the deterministic template over the stratified sample. No LLM, no credential."""
    import argparse
    import json
    import random
    import statistics
    from collections import Counter, defaultdict
    from datetime import UTC, datetime
    from pathlib import Path

    from biolit.domain.paper import Paper
    from biolit.domain.records import Cluster, ExtractedRecord
    from biolit.synth.template import render_cluster
    from biolit_evals._meta import git_sha
    from biolit_evals.synth_corpus import band_for, sample_clusters, sample_hash
    from biolit_evals.synth_metrics import (
        build_source_view,
        dcr,
        load_aliases,
        mesh_concepts,
        score_output,
    )

    parser = argparse.ArgumentParser(description="Gate A: score the template arm.")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--aliases", default=DEFAULT_ALIASES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--per-band", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required: it fixes which clusters are sampled. Never defaulted.",
    )
    args = parser.parse_args(argv)

    # Ruling 26. MERGED BY KEY, never concatenated. `cluster_papers` builds
    # `by_key: dict[str, set[str]]` and emits one Cluster per key, so keys are unique WITHIN a
    # run -- but this loop reads EIGHT runs over eight different queries, and two queries that
    # both surface the same chemical|disease pair yield two Clusters sharing a key with
    # different, possibly overlapping, paper sets. Concatenated, that one conceptual cluster is
    # sampled and scored twice, double-weighting it in a sample whose whole point is a
    # size-stratified 30, and landing in the wrong band besides, since neither copy carries the
    # full paper set. The union below is the same operation `cluster_papers` already performs
    # within a run, applied across runs. The merge count is printed because it changes band
    # membership and a reader is entitled to see by how much.
    merged: dict[str, set[str]] = defaultdict(set)
    records: dict[str, ExtractedRecord] = {}
    papers: dict[str, Paper] = {}
    n_raw = 0
    for path in sorted(Path(args.states).glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        for raw in state["clusters"]:
            n_raw += 1
            merged[raw["key"]].update(raw["paper_ids"])
        records.update({pid: ExtractedRecord(**r) for pid, r in state["extracted_records"].items()})
        papers.update({p["id"]: Paper(**p) for p in state["candidate_papers"]})
    clusters = [Cluster(key=key, paper_ids=sorted(pids)) for key, pids in sorted(merged.items())]
    print(
        f"{n_raw} raw clusters merged to {len(clusters)} by key "
        f"({n_raw - len(clusters)} duplicate keys), {len(papers)} papers from {args.states}"
    )

    sample = sample_clusters(clusters, rng=random.Random(args.seed), per_band=args.per_band)
    aliases = load_aliases(args.aliases)

    rows = []
    for cluster in sample:
        output = render_cluster(cluster, records, papers)
        score = score_output(output, cluster, records, papers, aliases)
        rows.append(
            {
                "key": cluster.key,
                "band": band_for(len(cluster.paper_ids)),
                "n_papers": len(cluster.paper_ids),
                "support_rate": score.support.rate,
                "unsupported": list(score.support.unsupported),
                "coverage_rate": score.coverage.rate,
                "dcr_rate": score.dcr.rate,
                "dcr_scorable": score.dcr.scorable,
                "dcr_indistinguishable": list(score.dcr.indistinguishable),
                # Ruling 19. `no_findings` is the whole point of Ruling 10 -- without it the
                # report cannot tell "the corpus carries duplicate findings" from "extraction
                # produced nothing", which mean opposite things. `lost` is the paper-level
                # detail §3.1's rule audits, and the itemisation both DCR bias notes in spec §2
                # name as their only mitigation.
                "dcr_no_findings": list(score.dcr.no_findings),
                "dcr_lost": list(score.dcr.lost),
                # Ruling 16. The token floor is pre-registered at 4 but its value is not
                # self-evident: the per-cluster set-difference already removes shared tokens,
                # so the floor's live effect is dropping short biomedical markers (IL6, TNF,
                # BMI). Logged at 3 as a sensitivity figure on the real sample; if the two
                # disagree materially that is reported, not hidden.
                "dcr_rate_floor3": dcr(output, cluster, records, min_token_len=3).rate,
                "compression": score.compression,
                # Ruling 32. `compression` returns 0.0 for a cluster with no extracted
                # findings -- a sentinel, not a measurement -- so the pooled mean needs
                # to exclude those rows for exactly the reason `dcr_mean` excludes
                # zero-scorable ones. Logged as a length so the exclusion is visible in
                # the row rather than inferred from a magic 0.0.
                "findings_chars": len(build_source_view(cluster, records, papers).findings_text),
                "hallucinated": list(score.hallucinated),
                "judgment_terms": list(score.judgment_terms),
                # Ruling 27. Both, always. `judgment_terms` carries the corpus's own
                # clinical vocabulary (the template scored 10/30 quoting "consistent
                # with rhabdomyolysis"); `judgment_volunteered` is what the arm added
                # and is construction-true at 0 for the template. Subtracting the source
                # alone would hide an arm writing "the findings are consistent across
                # studies" over a corpus that says "consistent with"; counting everything
                # alone drowns the signal. Reported together and itemised, per §5.1.
                "judgment_volunteered": list(score.judgment_volunteered),
                # Ruling 7 diagnostic, logged and never scored. `paper.journal` is part of the
                # source view, so a cluster published in a disease-named journal ("Cancer",
                # "Pain") has that concept permanently exempt from the hallucination check.
                # Keeping the journal is deliberate -- the template renders it, so excluding it
                # would flag the CONTROL arm -- but the exemption's real size must be measured
                # in the pilot rather than assumed. Counts clusters, not concepts.
                "journal_alias_concepts": sorted(
                    mesh_concepts(
                        "\n".join(papers[pid].journal or "" for pid in cluster.paper_ids),
                        aliases,
                    )
                ),
                "output": output,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def mean(field: str, subset: list[dict]) -> float | None:
        values = [r[field] for r in subset]
        return statistics.fmean(values) if values else None

    def compression_mean(subset: list[dict]) -> float | None:
        """Ruling 32. Ruling 17's argument, transferred to the other comparative axis -- it
        was made for DCR and never carried across, though it holds word for word. A cluster
        with no extracted findings has no denominator, so `compression` returns the 0.0
        sentinel its own docstring names; that value depends only on the corpus, never on the
        output, so it hands BOTH arms the same number and compresses the arm gap. Measured on
        a synthetic corpus that exercises the path: 1.108 with the sentinels averaged in
        against 1.846 without, a 40% distortion. Latent on the frozen sample, which has no
        such cluster, and that is precisely why it needed transferring rather than waiting."""
        values = [r["compression"] for r in subset if r["findings_chars"] > 0]
        return statistics.fmean(values) if values else None

    def dcr_mean(subset: list[dict]) -> float | None:
        """Ruling 17. A cluster with no scorable papers returns rate 1.0, and `scorable`
        depends only on the cluster and its records -- never on the output -- so such a
        cluster hands BOTH arms a perfect score no matter what either wrote. Averaging it in
        compresses the arm gap by exactly the zero-scorable fraction, on the one axis the gate
        decides by, in the direction of the pre-registered default outcome. Excluded here and
        counted separately, so the denominator is visible rather than inferred."""
        values = [r["dcr_rate"] for r in subset if r["dcr_scorable"] > 0]
        return statistics.fmean(values) if values else None

    per_band = {
        band: {
            "n": len([r for r in rows if r["band"] == band]),
            "support_rate": mean("support_rate", [r for r in rows if r["band"] == band]),
            "coverage_rate": mean("coverage_rate", [r for r in rows if r["band"] == band]),
            "dcr_rate": dcr_mean([r for r in rows if r["band"] == band]),
            "compression": compression_mean([r for r in rows if r["band"] == band]),
            # Ruling 19. Without these a reader sees a high `dcr_rate` on the large band and
            # cannot tell real retention from a collapsed denominator -- and the large band is
            # where sibling papers most often share vocabulary, so collapse concentrates
            # exactly where the gate most wants to discriminate.
            "dcr_scorable_papers": sum(r["dcr_scorable"] for r in rows if r["band"] == band),
            "dcr_zero_scorable_clusters": sum(
                1 for r in rows if r["band"] == band and r["dcr_scorable"] == 0
            ),
            "dcr_no_findings_papers": sum(
                len(r["dcr_no_findings"]) for r in rows if r["band"] == band
            ),
        }
        for band in ("small", "medium", "large")
    }
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "step": "synth_gate_a",
        "arm": "template",
        "seed": args.seed,
        "sample_hash": sample_hash(sample),
        "n_clusters": len(rows),
        "pooled": {
            "support_rate": mean("support_rate", rows),
            "coverage_rate": mean("coverage_rate", rows),
            "dcr_rate": dcr_mean(rows),
            "dcr_rate_floor3": statistics.fmean(
                [r["dcr_rate_floor3"] for r in rows if r["dcr_scorable"] > 0]
            )
            if any(r["dcr_scorable"] > 0 for r in rows)
            else None,
            "compression": compression_mean(rows),
        },
        "per_band": per_band,
        "hallucinated_total": sum(len(r["hallucinated"]) for r in rows),
        "judgment_term_clusters": sum(1 for r in rows if r["judgment_terms"]),
        "judgment_volunteered_clusters": sum(1 for r in rows if r["judgment_volunteered"]),
        "dcr_indistinguishable_papers": sum(len(r["dcr_indistinguishable"]) for r in rows),
        "dcr_no_findings_papers": sum(len(r["dcr_no_findings"]) for r in rows),
        "dcr_zero_scorable_clusters": sum(1 for r in rows if r["dcr_scorable"] == 0),
        "compression_excluded_clusters": sum(1 for r in rows if r["findings_chars"] == 0),
        "journal_alias_clusters": sum(1 for r in rows if r["journal_alias_concepts"]),
    }
    with Path(args.log).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    print(f"wrote {out_path} and logged to {args.log}")
    print(f"pooled: {entry['pooled']}")
    print(f"per band: {json.dumps(per_band, indent=2)}")
    print(f"clusters using judgment language: {entry['judgment_term_clusters']}/{len(rows)}")
    print(
        "clusters VOLUNTEERING judgment language: "
        f"{entry['judgment_volunteered_clusters']}/{len(rows)}"
    )
    print(f"papers with no distinguishing tokens: {entry['dcr_indistinguishable_papers']}")
    print(f"papers with no extracted findings: {entry['dcr_no_findings_papers']}")
    print(f"clusters excluded from DCR (0 scorable): {entry['dcr_zero_scorable_clusters']}")
    print(f"DCR at token floor 3 (sensitivity): {entry['pooled']['dcr_rate_floor3']}")
    print(f"clusters with an exempt journal concept: {entry['journal_alias_clusters']}/{len(rows)}")
    print(f"strata: {dict(sorted(Counter(r['band'] for r in rows).items()))}")


if __name__ == "__main__":  # pragma: no cover
    main()
