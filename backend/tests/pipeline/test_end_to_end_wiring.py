from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.cluster.pairing import SameSentencePairing
from biolit.domain.enums import EntityLabel, LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.pipeline.stages import (
    cluster_stage,
    critic_stub,
    entities_stage,
    records_stage,
    retrieve_stage,
    select_stage,
    synthesis_stage,
)
from biolit.query.concepts import QueryConcepts
from biolit.state.pipeline import PipelineState, StageStatus

NO_ACTIONS = PharmacologicalActions({})

ABSTRACT = "Metformin therapy was associated with acidosis in this cohort."


def _entities() -> list[Entity]:
    return [
        Entity(
            text="Metformin",
            label=EntityLabel.CHEMICAL,
            start=0,
            end=9,
            canonical_id="D008687",
            canonical_name="Metformin",
        ),
        Entity(
            text="acidosis",
            label=EntityLabel.DISEASE,
            start=38,
            end=46,
            canonical_id="D000138",
            canonical_name="Acidosis",
        ),
    ]


def _paper(pid: str, *, allowed: bool) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        abstract=ABSTRACT,
        text_type=TextType.abstract_only,
        license_tier=LicenseTier.open if allowed else LicenseTier.unknown,
        license="cc_by" if allowed else None,
        extraction_allowed=allowed,
    )


def test_the_whole_pipeline_runs_and_the_ledger_accounts_for_every_paper():
    """The wiring test. Two permitted papers cluster; one refused paper is accounted for.

    Asserts the LEDGER, not just the outputs -- the ledger is the deliverable, and it is
    the thing most likely to regress silently.
    """
    papers = [_paper("a", allowed=True), _paper("b", allowed=True), _paper("c", allowed=False)]
    state = PipelineState(question="does metformin cause acidosis?")
    state.candidate_papers = papers
    state.stages.append(retrieve_stage(["1", "2", "3"], papers))

    entities, entity_report = entities_stage(papers, extract=lambda text: _entities())
    state.stages.append(entity_report)

    outcome = records_stage(papers, entities)
    state.extracted_records = outcome.records
    state.stages.extend([outcome.licence, outcome.extract])

    clusters, cluster_report = cluster_stage(
        list(outcome.records.values()),
        texts={p.id: ABSTRACT for p in papers},
        pairing=SameSentencePairing(),
    )
    state.stages.append(cluster_report)

    # The query is CONSULTED here and nowhere else after esearch. Before this stage existed
    # `PipelineState.question` was write-only, and every cluster retrieval happened to
    # produce went into the answer regardless of what was asked.
    concepts = QueryConcepts(frozenset({"D008687"}), {"D008687": "metformin"}, ())
    clusters, select_report = select_stage(
        clusters, concepts, tree=MeshTree({}), actions=NO_ACTIONS
    )
    state.clusters = clusters
    state.stages.append(select_report)
    state.stages.append(critic_stub(len(clusters)))
    answer, synthesis_report = synthesis_stage(clusters, outcome.records, {p.id: p for p in papers})
    state.answer = answer
    state.stages.append(synthesis_report)

    assert set(state.extracted_records) == {"a", "b"}
    assert [c.paper_ids for c in state.clusters] == [["a", "b"]]

    by_name = {stage.name: stage for stage in state.stages}
    assert by_name["licence_gate"].dropped == {"licence_refused:none": 1}
    assert by_name["select"].n_in == 1 and by_name["select"].n_out == 1
    assert by_name["critic"].status is StageStatus.not_implemented
    # ADR-0019: synthesis is implemented now; the Critic remains retired.
    assert by_name["synthesis"].status is StageStatus.completed
    assert state.answer is not None
    assert "PMID a" in state.answer and "PMID b" in state.answer


def test_the_retired_critic_survives_the_json_dump_as_not_implemented():
    """The requirement most likely to regress silently, so it gets its own test: a machine
    reader of the dump must see not_implemented rather than an empty contradictions list.
    ADR-0019 implemented synthesis, so only the Critic is left to make this claim -- and it
    matters more now, not less: a dump carrying a real synthesis answer alongside an empty
    contradictions list is exactly where a reader might infer "no contradictions found"."""
    _, synthesis_report = synthesis_stage([], {}, {})
    state = PipelineState(question="q", stages=[critic_stub(0), synthesis_report])
    payload = state.model_dump(mode="json")
    statuses = {stage["name"]: stage["status"] for stage in payload["stages"]}
    assert statuses == {"critic": "not_implemented", "synthesis": "completed"}
    assert payload["contradictions"] == []
    assert payload["answer"] is None
