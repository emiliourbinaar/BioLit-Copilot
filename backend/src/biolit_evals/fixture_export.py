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

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.domain.paper import Paper
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

_WHITESPACE = re.compile(r"\s+")


def _normalise_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _iter_strings(value: object):
    """Walk a `model_dump()`-shaped structure, yielding every string leaf.

    Deliberately over the DECODED structure, not `model_dump_json()`'s escaped text: a
    leaked passage containing a `"`, a `\\`, or a newline round-trips through JSON escaping
    and would no longer equal itself as plain text there, which is exactly the gap a leak
    check must not have.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            yield from _iter_strings(item)


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
    run = FixtureRun(
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
    _assert_ledger_balances(run, slug=slug)
    _assert_no_papers_collapsed(run, state, slug=slug)
    _assert_no_leaked_text(run, state.candidate_papers, slug=slug)
    return run


def _assert_no_leaked_text(run: FixtureRun, papers: Sequence[Paper], *, slug: str) -> None:
    """⛔ THE SPEC'S §5 REQUIREMENT, and it belongs HERE, inside the pure projection, rather
    than only in `main()` -- a second caller of `project_run` must get the same defence. A
    fixture must be impossible to PRODUCE unsanitised, not merely checkable afterwards --
    "we'll check later" is the exact shape of DEF-0006. Content, not the literal word
    "abstract": a leak copies TEXT, not a field name.

    10-word windows at stride 5, which catches any leaked run of >= 14 consecutive words. Not
    exhaustive, and deliberately not claimed to be: the primary guarantee is structural --
    PaperStub has no abstract field at all -- and this is defence in depth behind it.

    Matched against decoded string values with whitespace normalised on both sides, so a
    leaked passage cannot dodge the check by way of JSON escaping (a `"` or `\\`) or by being
    re-wrapped across a newline somewhere between the source and this fixture.
    """
    blob = _normalise_whitespace(" ".join(_iter_strings(run.model_dump())))
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


def _assert_no_papers_collapsed(run: FixtureRun, state: PipelineState, *, slug: str) -> None:
    """Every DISTINCT retrieved paper must get a stub -- the projection may not lose one.

    ⚠️ COMPARED AGAINST DISTINCT IDS, NOT THE RAW PAPER COUNT, and the difference matters. An
    earlier version of this check required `len(run.papers) == len(state.candidate_papers)` and
    refused every run containing a duplicate DOI -- which is a real occurrence (DEF-0007), not
    an error in the projection. That version was wrong about what `papers` IS.

    `papers` is a LOOKUP TABLE keyed by `Paper.id`, not a count. The count of record lives in
    the stage ledger, where `retrieve` reports 58 and `licence_gate` now subtracts the collapse
    explicitly as `duplicate_paper_id` (DEF-0007). A site that counts `len(papers)` to say "58
    retrieved" is reading the wrong field; it should read the ledger, which is why the ledger
    balances.

    What this still catches: a stub silently missing for an id that some cluster cites, or a
    projection bug that drops a paper for any reason other than sharing an id.
    """
    distinct = {paper.id for paper in state.candidate_papers}
    if len(run.papers) != len(distinct):
        raise RuntimeError(
            f"{slug}: {len(distinct)} distinct paper ids produced only {len(run.papers)} "
            "stubs -- the projection lost a paper for a reason other than a duplicate id. "
            "Refusing to write a fixture that cannot attribute everything it may cite."
        )


def _assert_ledger_balances(run: FixtureRun, *, slug: str) -> None:
    """`StageReport`'s docstring promises that where `unit_in == unit_out`, the ledger is
    checkable: `n_in - sum(dropped) == n_out`. A fixture that fails this is a false claim on
    a public page (measured: the committed `statins-rhabdomyolysis` fixture once said
    `n_in=58, dropped=17, n_out=40` -- 58-17=41, not 40 -- because a duplicate `Paper.id`
    collapsed two retrieved papers into one dict entry downstream). Raise loudly rather than
    let that ship silently again.
    """
    for stage in run.stages:
        if stage.unit_in != stage.unit_out:
            continue
        expected = stage.n_in - sum(stage.dropped.values())
        if expected != stage.n_out:
            raise RuntimeError(
                f"{slug}: stage {stage.name!r} ledger does not balance: "
                f"n_in={stage.n_in} - sum(dropped)={sum(stage.dropped.values())} "
                f"= {expected}, but n_out={stage.n_out}. Refusing to write a fixture whose "
                "own ledger fails the arithmetic its docstring promises."
            )


#: The four runs, and the finding each exists to show. Slugs are stable: the frontend routes
#: on them.
FEATURED: dict[str, str] = {
    "statins-rhabdomyolysis": "statins and rhabdomyolysis",
    "isotretinoin-depression": "isotretinoin and depression",
    "cisplatin-nephrotoxicity": "cisplatin nephrotoxicity",
    "metformin-lactic-acidosis": "metformin and lactic acidosis",
}

#: ⚠️ ANCHORED ON __file__, NOT THE CWD. Resolved relatively, running the generator from the
#: repo root instead of `backend/` wrote fixtures OUTSIDE the repo and still printed success --
#: a silent-wrong-place failure, which is the shape this project keeps finding and refusing.
#: src/biolit_evals/fixture_export.py -> parents: biolit_evals, src, backend, <repo root>.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = str(_REPO_ROOT / "frontend" / "src" / "fixtures")
#: Same reasoning. A missing labels file RAISES rather than silently producing fixtures whose
#: every `label` is absent -- the site would render an unlabelled run as though it had no
#: annotation, which is a different claim from "the annotation was not loaded".
LABELS_PATH = _REPO_ROOT / "backend" / "evals" / "gold" / "relevance_labels.jsonl"
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
    if not LABELS_PATH.exists():
        raise RuntimeError(
            f"fixture_export: relevance labels not found at {LABELS_PATH}. Refusing to "
            "generate: every cluster would ship with `label` absent, and the site cannot "
            "distinguish that from a run nobody annotated."
        )
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
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

        # `project_run` is where the leak check, the ledger-balance check and the
        # paper-count check now live (moved from here): a second caller of `project_run`
        # must get the same defences, not just this function.
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
        blob = run.model_dump_json(indent=2)

        path = out_dir / f"{slug}.json"
        path.write_text(blob, encoding="utf-8")
        print(f"{slug}: {len(run.clusters)} clusters, {len(run.papers)} papers -> {path}")

    for slug in slugs:
        asyncio.run(build(slug))


if __name__ == "__main__":
    main()
