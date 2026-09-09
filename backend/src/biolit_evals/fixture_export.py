"""Project a finished `PipelineState` into a publishable, licence-sanitised fixture.

⚠️ THIS IS NOT A WRAPPER AROUND `--json-out`, AND MUST NEVER BECOME ONE. `--json-out` is the
path DEF-0006 is filed against: it writes refused papers' verbatim abstracts to disk. Driving
generation through it would create the unsafe file first and sanitise second, and the unsafe
file must never exist -- not merely never be committed. `main()` therefore calls the stage
functions in-process and hands the state straight to `project_run`.

`project_run` is pure: no network, no filesystem, no clock. Everything variable is a
parameter, so the test can construct a state containing a sentinel string and assert the
sentinel cannot reach the output by any route.
"""

from collections.abc import Mapping, Sequence

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.query.concepts import QueryConcepts
from biolit.query.ranking import relevance_score
from biolit.state.pipeline import PipelineState
from biolit_evals.fixture_models import (
    SCHEMA_VERSION,
    FixtureCluster,
    FixtureFinding,
    FixtureRun,
    PaperStub,
)
from biolit_evals.fixture_pin import source_pin


def project_run(
    state: PipelineState,
    *,
    slug: str,
    concepts: QueryConcepts,
    tree: MeshTree,
    actions: PharmacologicalActions,
    names: Mapping[str, str],
    labels: Mapping[str, str],
    findings: Sequence[FixtureFinding],
    generated_at: str,
) -> FixtureRun:
    """Sanitised projection. Abstracts cannot survive it, because the schema has no field."""
    papers = {
        paper.id: PaperStub(
            title=paper.title,
            journal=paper.journal,
            year=paper.year,
            doi=paper.doi,
            pmid=paper.pmid,
            license=paper.license,
            license_tier=str(paper.license_tier),
            extraction_allowed=paper.extraction_allowed,
        )
        for paper in state.candidate_papers
    }
    clusters = []
    for rank, cluster in enumerate(state.clusters, start=1):
        matched, proximity = relevance_score(cluster, concepts, tree=tree, actions=actions)
        sides = cluster.key.split("|")
        # inf is a sort-time stand-in for "no shared tree placement"; JSON cannot carry it and
        # a float-typed field cannot reload the `null` it becomes. Restore the category.
        finite = None if proximity == float("inf") else proximity
        clusters.append(
            FixtureCluster(
                key=cluster.key,
                concept_names=[names.get(side, side) for side in sides],
                paper_ids=list(cluster.paper_ids),
                rank=rank,
                matched=matched,
                proximity=finite,
                label=labels.get(cluster.key),
            )
        )
    return FixtureRun(
        schema_version=SCHEMA_VERSION,
        slug=slug,
        query=state.question,
        generated_at=generated_at,
        source_pin=source_pin(),
        stages=list(state.stages),
        clusters=clusters,
        answer=state.answer or "",
        papers=papers,
        findings=list(findings),
    )


#: The four runs, and the finding each exists to show. Slugs are stable: the frontend routes
#: on them.
FEATURED: dict[str, str] = {
    "statins-rhabdomyolysis": "statins and rhabdomyolysis",
    "isotretinoin-depression": "isotretinoin and depression",
    "cisplatin-nephrotoxicity": "cisplatin nephrotoxicity",
    "metformin-lactic-acidosis": "metformin and lactic acidosis",
}

DEFAULT_OUT = "../frontend/src/fixtures"
#: 60, matching the frozen corpus this project's published numbers come from -- measured,
#: not guessed: its eight runs retrieved 58-60 papers each. An earlier draft of this plan
#: said 40 while asserting a cluster count taken from a 58-paper run, which is not a check
#: a 40-paper run can pass.
DEFAULT_MAX_PAPERS = 60


def main(argv: list[str] | None = None) -> None:
    """Regenerate fixtures by running the real pipeline in-process.

    ⚠️ IN-PROCESS, NEVER THROUGH `--json-out`. That path writes refused papers' abstracts to
    disk (DEF-0006); the unsafe artifact must never exist, not merely never be committed.

    `main()` gets no direct unit test per project convention; `project_run` and `source_pin`
    are tested in their own modules.
    """
    import argparse
    import asyncio
    import gzip
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    import httpx

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.clients.pubmed import PubMedClient
    from biolit.cluster.pairing import SameSentencePairing
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit.pipeline.stages import (
        cluster_stage,
        critic_stub,
        entities_stage,
        records_stage,
        retrieve_stage,
        select_stage,
        synthesis_stage,
    )
    from biolit.query.concepts import resolve_query_concepts
    from biolit.state.pipeline import PipelineState

    parser = argparse.ArgumentParser(description="Regenerate evidence-viewer fixtures.")
    parser.add_argument("--slug", default=None, help="one slug, or all of FEATURED")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--max-papers", type=int, default=DEFAULT_MAX_PAPERS)
    args = parser.parse_args(argv)

    slugs = [args.slug] if args.slug else list(FEATURED)
    unknown = [s for s in slugs if s not in FEATURED]
    if unknown:
        raise SystemExit(f"unknown slug(s): {unknown}; known: {sorted(FEATURED)}")

    settings = get_settings()
    tree = MeshTree.from_artifact(settings.mesh_tree_artifact_path)
    actions = PharmacologicalActions.from_artifact(settings.mesh_actions_artifact_path)
    dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
    linker = DictionaryLinker(dictionary)
    ner = NerModel.load(settings)

    names: dict[str, str] = {}
    with gzip.open(settings.mesh_artifact_path, "rt", encoding="utf-8") as handle:
        for _surface, entries in json.load(handle).items():
            for concept_id, name, _exact in entries:
                names.setdefault(concept_id, name)

    labels: dict[str, dict[str, str]] = {}
    labels_path = Path("evals/gold/relevance_labels.jsonl")
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row["is_distractor"]:
                labels.setdefault(row["query"], {})[row["cluster_key"]] = row["label"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    async def build(slug: str) -> None:
        query = FEATURED[slug]
        async with httpx.AsyncClient(timeout=60) as http:
            client = PubMedClient(http, settings)
            found = await client.esearch_detailed(query, retmax=args.max_papers)
            papers = await client.efetch(found.pmids)

        state = PipelineState(question=query)
        state.candidate_papers = papers
        state.stages.append(retrieve_stage(found.pmids, papers))

        def extract(text: str):
            found_entities = extract_entities(
                text, ner, score_threshold=settings.ner_score_threshold
            )
            return canonicalize(found_entities, text, linker=linker)

        entities_by_paper, entity_report = entities_stage(papers, extract=extract)
        state.stages.append(entity_report)

        outcome = records_stage(papers, entities_by_paper)
        state.extracted_records = outcome.records
        # TWO reports, not one: the licence gate and the extractor are separate ledger rows,
        # and the site renders `licence_refused` as its own line.
        state.stages.extend([outcome.licence, outcome.extract])

        texts = {paper.id: paper.abstract or "" for paper in papers}
        clusters, cluster_report = cluster_stage(
            list(outcome.records.values()), texts=texts, pairing=SameSentencePairing()
        )
        state.stages.append(cluster_report)

        concepts = resolve_query_concepts(found.concept_terms, lookup=linker.link)
        clusters, select_report = select_stage(clusters, concepts, tree=tree, actions=actions)
        state.clusters = clusters
        state.stages.append(select_report)

        state.stages.append(critic_stub(len(clusters)))
        answer, synthesis_report = synthesis_stage(
            clusters, outcome.records, {paper.id: paper for paper in papers}
        )
        state.answer = answer
        state.stages.append(synthesis_report)

        run = project_run(
            state,
            slug=slug,
            concepts=concepts,
            tree=tree,
            actions=actions,
            names=names,
            labels=labels.get(query, {}),
            findings=[],
            generated_at=datetime.now(UTC).isoformat(),
        )
        # ⛔ THE SPEC'S §5 REQUIREMENT, and it belongs HERE rather than in a verification step
        # someone can forget to run. A fixture must be impossible to WRITE unsanitised, not
        # merely checkable afterwards -- "we'll check later" is the exact shape of DEF-0006.
        # Content, not the literal word "abstract": a leak copies TEXT, not a field name.
        # 10-word windows at stride 5, which catches any leaked run of >= 14 consecutive
        # words. Not exhaustive, and deliberately not claimed to be: the primary guarantee is
        # structural -- PaperStub has no abstract field at all -- and this is defence in depth
        # behind it.
        blob = run.model_dump_json(indent=2)
        for paper in papers:
            if paper.extraction_allowed or not paper.abstract:
                continue
            words = paper.abstract.split()
            shingles = [" ".join(words[i : i + 10]) for i in range(0, max(len(words) - 10, 1), 5)]
            leaked = [sh for sh in shingles if sh and sh in blob]
            if leaked:
                raise RuntimeError(
                    f"{slug}: REFUSED paper {paper.id} leaked text into the fixture: "
                    f"{leaked[0]!r}. Refusing to write. This is DEF-0006 reaching a public "
                    "page; fix the projection rather than this check."
                )

        path = out_dir / f"{slug}.json"
        path.write_text(blob, encoding="utf-8")
        print(f"{slug}: {len(run.clusters)} clusters, {len(run.papers)} papers -> {path}")

    for slug in slugs:
        asyncio.run(build(slug))


if __name__ == "__main__":
    main()
