from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.cluster.pairing import SameSentencePairing
from biolit.domain.enums import EntityLabel, LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, Entity, ExtractedRecord
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
from biolit.state.pipeline import StageStatus

NO_ACTIONS = PharmacologicalActions({})


def _paper(pid: str, abstract: str | None, *, allowed: bool) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        abstract=abstract,
        text_type=TextType.abstract_only,
        license_tier=LicenseTier.open if allowed else LicenseTier.unknown,
        license="cc_by" if allowed else None,
        extraction_allowed=allowed,
    )


def test_retrieve_stage_counts_pmids_that_produced_no_paper_and_papers_with_no_abstract():
    papers = [
        _paper("a", "Metformin causes acidosis.", allowed=True),
        _paper("b", None, allowed=True),
    ]
    report = retrieve_stage(["1", "2", "3"], papers)
    assert report.status is StageStatus.completed
    assert report.n_in == 3
    assert report.n_out == 2
    # The two kinds side by side, which is the whole distinction: one pmid returned no
    # article and is genuinely gone; the abstract-less paper is still in `papers`.
    assert report.dropped == {"no_article_returned": 1}
    assert report.noted == {"no_abstract": 1}


def test_entities_stage_counts_unlinked_entities_without_discarding_them():
    """NIL entities are COUNTED, not dropped. `SameSentenceAsEntitiesExtractor` already
    fails closed on an unlinked entity; removing them here would double-count the same loss
    and make the ledger disagree with what the extractor actually saw."""
    linked = Entity(
        text="metformin",
        label=EntityLabel.CHEMICAL,
        start=0,
        end=9,
        canonical_id="D008687",
        canonical_name="Metformin",
    )
    nil = Entity(text="wibble", label=EntityLabel.DISEASE, start=10, end=16)

    entities, report = entities_stage(
        [_paper("a", "metformin wibble", allowed=True)],
        extract=lambda text: [linked, nil],
    )
    assert entities["a"] == [linked, nil]
    assert report.dropped == {}
    assert report.noted["entity_unlinked"] == 1
    assert (report.unit_in, report.unit_out) == ("papers", "entities")
    assert report.n_out == 2


def test_records_stage_reports_the_licence_refusal_broken_out_by_licence():
    """The licence gate is `build_record`, which returns None for a refused paper. The
    ledger must break refusals out by the licence actually seen, because on real data
    roughly half of retrieved papers are correctly refused and a bare count reads as a bug.
    """
    papers = [
        _paper("ok", "Metformin causes acidosis.", allowed=True),
        _paper("no", "Metformin causes acidosis.", allowed=False),
    ]
    outcome = records_stage(papers, {"ok": [], "no": []})
    assert set(outcome.records) == {"ok"}
    assert outcome.licence.n_in == 2
    assert outcome.licence.n_out == 1
    assert outcome.licence.dropped == {"licence_refused:none": 1}


def test_records_stage_counts_records_that_yielded_no_findings():
    """A permitted paper with no chemical+disease sentence produces an empty record. That
    is a real outcome, not an error -- but it must be visible, or an empty final result
    looks like a crash."""
    paper = _paper("ok", "This abstract mentions nothing linkable.", allowed=True)
    outcome = records_stage([paper], {"ok": []})
    assert outcome.extract.dropped == {}
    assert outcome.extract.noted["zero_findings"] == 1
    assert outcome.extract.n_out == 1


def test_cluster_stage_separates_singleton_keys_from_papers_that_produced_no_pair():
    """Two different drops with the same visible effect (no cluster) and different causes.
    Collapsing them would hide which one is happening on a small result set."""
    shared = "Metformin causes acidosis."
    entities = [
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
            start=17,
            end=25,
            canonical_id="D000138",
            canonical_name="Acidosis",
        ),
    ]
    records = [
        ExtractedRecord(paper_id="a", entities=entities),
        ExtractedRecord(paper_id="b", entities=entities),
        ExtractedRecord(paper_id="c", entities=[]),
    ]
    texts = {"a": shared, "b": shared, "c": "Nothing here."}

    clusters, report = cluster_stage(records, texts=texts, pairing=SameSentencePairing())

    assert [c.paper_ids for c in clusters] == [["a", "b"]]
    assert report.noted.get("no_pairs") == 1
    assert report.dropped == {"no_cluster": 1}
    assert report.n_out == 1


def test_cluster_stage_counts_a_key_held_by_only_one_paper_as_a_singleton_drop():
    entities_a = [
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
            start=17,
            end=25,
            canonical_id="D000138",
            canonical_name="Acidosis",
        ),
    ]
    records = [ExtractedRecord(paper_id="a", entities=entities_a)]
    clusters, report = cluster_stage(
        records, texts={"a": "Metformin causes acidosis."}, pairing=SameSentencePairing()
    )
    assert clusters == []
    assert report.noted["singleton_key"] == 1
    assert (report.unit_in, report.unit_out) == ("records", "clusters")


def test_critic_stub_reports_not_implemented_and_points_at_the_adr():
    """Not a placeholder result. A stub that returned any ContradictionFinding at all would
    be worse than an empty list, because it manufactures a result where none exists."""
    report = critic_stub(n_clusters=4)
    assert report.status is StageStatus.not_implemented
    assert report.n_in == 4
    assert report.n_out == 0
    assert "ADR-0017" in (report.note or "")


def test_synthesis_stage_renders_every_cluster_and_reports_completed():
    """ADR-0019. Synthesis is no longer a stub: the deterministic template IS the stage.
    Gate A could not demonstrate an LLM advantage over it -- not because the LLM lost, but
    because the checkable axes could not tell "better" from "shorter", so no arm was ever
    bought. The template ships as the only option needing no unjustifiable judgment call.
    """
    from biolit.domain.records import Cluster, Finding

    papers = {
        "p1": _paper("p1", "irrelevant", allowed=True),
        "p2": _paper("p2", "irrelevant", allowed=True),
    }
    records = {
        pid: ExtractedRecord(
            paper_id=pid,
            key_findings=[Finding(text=f"Finding for {pid}.", start=0, end=1, sentence_index=0)],
        )
        for pid in papers
    }
    clusters = [Cluster(key="MESH:D1|MESH:D2", paper_ids=["p1", "p2"])]

    answer, report = synthesis_stage(clusters, records, papers)

    assert report.status is StageStatus.completed
    assert report.n_in == 1
    assert report.n_out == 1
    assert answer is not None
    assert "Finding for p1." in answer
    assert "Finding for p2." in answer


def test_synthesis_stage_returns_no_answer_when_there_is_nothing_to_synthesise():
    """Zero clusters is the ordinary outcome of a narrow query, not an error. `answer` stays
    None rather than becoming an empty string, so a caller can tell "nothing to say" from
    "said nothing"."""
    answer, report = synthesis_stage([], {}, {})

    assert answer is None
    assert report.status is StageStatus.completed
    assert report.n_out == 0


def _concepts(*ids: str) -> QueryConcepts:
    return QueryConcepts(ids=frozenset(ids), evidence={i: i for i in ids}, unresolved=())


def test_select_stage_drops_clusters_that_share_no_concept_with_the_query():
    """The gap this stage closes: before it, `synthesis_stage` rendered every cluster
    retrieval happened to produce, and the query string was never consulted again after
    `esearch`. On the frozen metformin run that shipped `Creatinine | Hyperkalemia` and
    `Potassium | Acute Kidney Injury` inside the answer to "metformin and lactic acidosis".
    """
    on_query = Cluster(key="MESH:D008687|MESH:D000140", paper_ids=["a", "b"])
    off_query = Cluster(key="MESH:D003404|MESH:D006947", paper_ids=["c", "d"])

    kept, report = select_stage(
        [on_query, off_query], _concepts("MESH:D008687"), tree=MeshTree({}), actions=NO_ACTIONS
    )

    assert kept == [on_query]
    assert report.status is StageStatus.completed
    assert (report.n_in, report.n_out) == (2, 1)
    assert (report.unit_in, report.unit_out) == ("clusters", "clusters")
    assert report.dropped == {"off_query": 1}


def test_select_stage_records_no_drop_key_when_every_cluster_is_on_query():
    """An absent key, not a zero. Every other stage in this module omits a drop it did not
    make, and a `{"off_query": 0}` row would read in the rendered ledger as a drop that
    happened to be empty."""
    cluster = Cluster(key="MESH:D008687|MESH:D000140", paper_ids=["a", "b"])

    kept, report = select_stage(
        [cluster], _concepts("MESH:D008687"), tree=MeshTree({}), actions=NO_ACTIONS
    )

    assert kept == [cluster]
    assert report.dropped == {}


def test_select_stage_fails_open_when_the_query_resolved_to_no_concept_at_all():
    """FAIL-OPEN, and the reason it must be open rather than closed: filtering on an empty
    concept set drops everything, so a query NCBI could not translate would return no answer
    at all rather than an unfiltered one. Returning the clusters unfiltered is a strictly
    smaller failure than returning nothing, but it is still a failure, so it is `noted` --
    a reader must be able to tell an unfiltered run from a filtered one that kept everything.
    """
    clusters = [
        Cluster(key="MESH:D008687|MESH:D000140", paper_ids=["a", "b"]),
        Cluster(key="MESH:D003404|MESH:D006947", paper_ids=["c", "d"]),
    ]

    kept, report = select_stage(
        clusters, QueryConcepts(frozenset(), {}, ("wibble",)), tree=MeshTree({}), actions=NO_ACTIONS
    )

    assert kept == clusters
    assert report.dropped == {}
    assert report.noted == {"query_unlinked": 1}
    assert "unfiltered" in (report.note or "").lower()


def test_select_stage_does_not_rescue_a_query_that_filters_down_to_nothing():
    """A DELIBERATE non-rescue, recorded because the opposite is tempting. Falling back to
    the unfiltered list whenever the filter empties would be a hidden threshold of exactly
    the kind this project forbids, and it would mask the one case a reader most needs to
    see. `synthesis_stage` already renders an empty cluster list as `answer is None`, which
    is a truthful "nothing here answers that", and the ledger shows the whole drop.
    """
    off_query = Cluster(key="MESH:D003404|MESH:D006947", paper_ids=["c", "d"])

    kept, report = select_stage(
        [off_query], _concepts("MESH:D008687"), tree=MeshTree({}), actions=NO_ACTIONS
    )

    assert kept == []
    assert report.n_out == 0
    assert report.dropped == {"off_query": 1}


def test_synthesis_stage_quotes_a_paper_once_across_clusters_and_counts_the_repeats():
    """A4 at the join. A paper carrying two chemical|disease pairs is a member of two
    clusters and must appear in both -- dropping it from one would misreport the cluster --
    but its sentences belong in the answer once. The count is `noted`, not `dropped`:
    nothing is removed from the pipeline, only from the rendered text."""
    from biolit.domain.records import Finding

    papers = {pid: _paper(pid, "irrelevant", allowed=True) for pid in ("p1", "p2")}
    records = {
        pid: ExtractedRecord(
            paper_id=pid,
            key_findings=[Finding(text=f"Finding for {pid}.", start=0, end=1, sentence_index=0)],
        )
        for pid in papers
    }
    clusters = [
        Cluster(key="MESH:D1|MESH:D2", paper_ids=["p1", "p2"]),
        Cluster(key="MESH:D1|MESH:D3", paper_ids=["p1"]),
    ]

    answer, report = synthesis_stage(clusters, records, papers)

    assert answer is not None
    assert answer.count("Finding for p1.") == 1
    assert answer.count("Finding for p2.") == 1
    assert report.noted == {"repeat_appearance": 1}
    assert report.n_out == 2


def test_synthesis_stage_notes_no_repeats_when_every_paper_appears_once():
    cluster = Cluster(key="MESH:D1|MESH:D2", paper_ids=["p1"])
    papers = {"p1": _paper("p1", "irrelevant", allowed=True)}
    records = {"p1": ExtractedRecord(paper_id="p1")}

    _answer, report = synthesis_stage([cluster], records, papers)

    assert report.noted == {}


def test_select_stage_reports_selection_and_ordering_as_two_distinguishable_facts():
    """ADR-0020 folds ordering into this stage because it uses the identical signal, but the
    two remain separate decisions and the ledger must not blur them: one says what was
    REMOVED and on what basis, the other says the survivors were REORDERED. A single vague
    line would make it impossible to tell from the record whether ordering happened at all.
    """
    on_query = Cluster(key="MESH:D008687|MESH:D000140", paper_ids=["a"])
    off_query = Cluster(key="MESH:D003404|MESH:D006947", paper_ids=["c"])

    _kept, report = select_stage(
        [on_query, off_query], _concepts("MESH:D008687"), tree=MeshTree({}), actions=NO_ACTIONS
    )

    note = report.note or ""
    assert "kept 1 of 2" in note
    assert "concept overlap" in note
    assert "ordered by relevance" in note
    assert note.count(";") >= 1, "the two facts must be separately readable"


def test_select_stage_orders_the_clusters_it_keeps_by_relevance():
    """The whole point of ADR-0020. Before it, the lead was whichever survivor had the
    alphabetically smallest MeSH id, which is uncorrelated with relevance by construction."""
    alphabetically_first = Cluster(key="MESH:AAA|MESH:BBB", paper_ids=["a"])
    on_query = Cluster(key="MESH:QC|MESH:QD", paper_ids=["b"])
    tree = MeshTree({"MESH:QC": ["D01.1"], "MESH:AAA": ["D01.9"]})

    kept, _report = select_stage(
        [alphabetically_first, on_query],
        _concepts("MESH:QC", "MESH:QD"),
        tree=tree,
        actions=NO_ACTIONS,
    )

    assert kept[0] is on_query


def test_select_stage_says_it_did_not_reorder_when_the_query_resolved_to_nothing():
    """Fail-open must be honest about BOTH halves. With no concepts every cluster scores
    identically, so nothing is reordered -- and a note claiming relevance ordering would
    describe a sort that never consulted the query."""
    clusters = [Cluster(key="MESH:D008687|MESH:D000140", paper_ids=["a"])]

    _kept, report = select_stage(
        clusters, QueryConcepts(frozenset(), {}, ("wibble",)), tree=MeshTree({}), actions=NO_ACTIONS
    )

    note = report.note or ""
    assert "unfiltered" in note.lower()
    assert "not reordered" in note.lower()


def test_select_stage_keeps_a_cluster_that_matches_only_through_its_pharmacological_class():
    """ADR-0022 end to end, and the shape DEF-0002 named on the chemical side. "statins and
    rhabdomyolysis" resolves to the CLASS; the clusters carry a MEMBER, and MeSH files members
    by chemical structure (D03/D10) and classes by action (D27), so no tree walk connects them.

    The ledger must say which rule kept them. A reader who sees `Atorvastatin` in the answer to
    a question that never named Atorvastatin has no way to tell a class match from a linking
    bug unless the note distinguishes them.
    """
    member = Cluster(key="MESH:D000069059|MESH:D012206", paper_ids=["a"])
    off_query = Cluster(key="MESH:D003404|MESH:D006947", paper_ids=["b"])
    actions = PharmacologicalActions({"MESH:D000069059": ["MESH:D019161"]})

    kept, report = select_stage(
        [member, off_query], _concepts("MESH:D019161"), tree=MeshTree({}), actions=actions
    )

    assert kept == [member]
    assert report.dropped == {"off_query": 1}
    assert "pharmacological" in (report.note or "").lower()
