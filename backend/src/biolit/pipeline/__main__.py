"""CLI: run a query through the real Phase 1-5 components and print the stage ledger.

No LLM calls and no paid API calls. Every component is either local (the NER checkpoint,
the MeSH dictionary) or a free NCBI endpoint, so there is no pricing step and no
authorization gate -- there is nothing to authorize.

`main()` gets no direct unit test per project convention; every function it calls is
tested in its own module.
"""


def main(argv: list[str] | None = None) -> None:
    # Heavy imports local to main, same pattern as end_to_end.main (E402 exemption).
    import argparse
    import asyncio
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
    from biolit.pipeline.report import render_report
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

    parser = argparse.ArgumentParser(description="Run a query through the BioLit pipeline.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()

    async def retrieve():
        async with httpx.AsyncClient(timeout=60) as http:
            client = PubMedClient(http, settings)
            found = await client.esearch_detailed(args.query, retmax=args.max_papers)
            return found, await client.efetch(found.pmids)

    found, papers = asyncio.run(retrieve())
    pmids = found.pmids

    state = PipelineState(question=args.query)
    state.candidate_papers = papers
    state.stages.append(retrieve_stage(pmids, papers))

    model = NerModel.load(settings)
    linker = DictionaryLinker(MeshDictionary.from_artifact(settings.mesh_artifact_path))

    def extract(text: str):
        entities = extract_entities(text, model, score_threshold=settings.ner_score_threshold)
        return canonicalize(entities, text, linker=linker)

    entities_by_paper, entity_report = entities_stage(papers, extract=extract)
    state.stages.append(entity_report)

    outcome = records_stage(papers, entities_by_paper)
    state.extracted_records = outcome.records
    state.stages.extend([outcome.licence, outcome.extract])

    texts = {paper.id: paper.abstract or "" for paper in papers}
    clusters, cluster_report = cluster_stage(
        list(outcome.records.values()), texts=texts, pairing=SameSentencePairing()
    )
    state.clusters = clusters
    state.stages.append(cluster_report)

    # A3 + A1: NCBI translated the question into MeSH terms on the search request that
    # already happened; the dictionary turns those terms into concept ids, and the stage
    # keeps clusters that share one. Free, deterministic, no LLM.
    concepts = resolve_query_concepts(found.concept_terms, lookup=linker.link)
    clusters, select_report = select_stage(clusters, concepts)
    state.clusters = clusters
    state.stages.append(select_report)

    state.stages.append(critic_stub(len(clusters)))
    answer, synthesis_report = synthesis_stage(
        clusters, outcome.records, {paper.id: paper for paper in papers}
    )
    state.answer = answer
    state.stages.append(synthesis_report)

    print(f"query: {args.query!r}\n")
    print(render_report(state.stages))
    for cluster in clusters:
        print(f"\ncluster {cluster.key}: {', '.join(cluster.paper_ids)}")

    if answer:
        print(f"\n{answer}")

    if args.json_out:
        Path(args.json_out).write_text(state.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
