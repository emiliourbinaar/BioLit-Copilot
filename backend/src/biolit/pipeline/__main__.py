"""CLI: run a query through the real Phase 1-5 components and print the stage ledger.

No LLM calls and no paid API calls. Every component is either local (the NER checkpoint,
the MeSH dictionary) or a free NCBI endpoint, so there is no pricing step and no
authorization gate -- there is nothing to authorize.

`main()` gets no direct unit test per project convention; every function it calls is
tested in its own module. `emit_run` is factored out of it precisely so the persist-before-
display ordering below CAN be tested, because that ordering is a correctness property rather
than presentation.
"""

import sys
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from biolit.domain.records import Cluster
    from biolit.state.pipeline import PipelineState


def _write(out: "TextIO", text: str) -> None:
    """Write, degrading unencodable characters rather than failing the run.

    A Windows console is cp1252, and biomedical answers carry `≥`, `μ`, `α` and en dashes as a
    matter of course. Losing a completed run because its console cannot draw one glyph is not
    a defensible trade, and `errors="replace"` is what every other terminal-facing tool does.
    """
    try:
        out.write(text)
    except UnicodeEncodeError:
        encoding = getattr(out, "encoding", None) or "ascii"
        out.write(text.encode(encoding, "replace").decode(encoding))


def emit_run(
    state: "PipelineState",
    *,
    query: str,
    clusters: "list[Cluster]",
    json_out: str | None,
    out: "TextIO | None" = None,
) -> None:
    """PERSIST THE RUN, THEN DISPLAY IT. The order is the point.

    ⚠️ It used to be the other way round, and that discarded completed runs. Everything
    expensive -- retrieval, NER, the licence gate, extraction, clustering, selection -- is
    finished by the time this is called, so anything that can fail here must not be able to
    lose it. Measured on 2026-09-07: printing an answer containing `≥` to a cp1252 console
    raised before `--json-out` was reached and threw away 5 of 11 runs, which read as NCBI
    rate limiting until one was rerun with stderr visible.

    Display is best-effort and persistence is not. `_write` degrades a character it cannot
    encode; the artifact is byte-exact UTF-8 regardless of what the terminal can show.
    """
    from pathlib import Path

    from biolit.pipeline.report import render_report

    if json_out:
        Path(json_out).write_text(state.model_dump_json(indent=2), encoding="utf-8")

    stream: Any = out if out is not None else sys.stdout
    _write(stream, f"query: {query!r}\n\n")
    _write(stream, render_report(state.stages) + "\n")
    for cluster in clusters:
        _write(stream, f"\ncluster {cluster.key}: {', '.join(cluster.paper_ids)}\n")
    if state.answer:
        _write(stream, f"\n{state.answer}\n")
    if json_out:
        _write(stream, f"\nwrote {json_out}\n")


def main(argv: list[str] | None = None) -> None:
    # Heavy imports local to main, same pattern as end_to_end.main (E402 exemption).
    import argparse
    import asyncio

    import httpx

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.canon.mesh_actions import PharmacologicalActions
    from biolit.canon.mesh_tree import MeshTree
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
    clusters, select_report = select_stage(
        clusters,
        concepts,
        tree=MeshTree.from_artifact(settings.mesh_tree_artifact_path),
        actions=PharmacologicalActions.from_artifact(settings.mesh_actions_artifact_path),
    )
    state.clusters = clusters
    state.stages.append(select_report)

    state.stages.append(critic_stub(len(clusters)))
    answer, synthesis_report = synthesis_stage(
        clusters, outcome.records, {paper.id: paper for paper in papers}
    )
    state.answer = answer
    state.stages.append(synthesis_report)

    emit_run(state, query=args.query, clusters=clusters, json_out=args.json_out)


if __name__ == "__main__":
    main()
