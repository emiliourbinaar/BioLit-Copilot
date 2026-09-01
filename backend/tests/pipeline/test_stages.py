from biolit.cluster.pairing import SameSentencePairing
from biolit.domain.enums import EntityLabel, LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity, ExtractedRecord
from biolit.pipeline.stages import (
    cluster_stage,
    critic_stub,
    entities_stage,
    records_stage,
    retrieve_stage,
    synthesis_stub,
)
from biolit.state.pipeline import StageStatus


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
    assert report.dropped["no_abstract"] == 1


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
    assert report.dropped["entity_unlinked"] == 1
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
    assert outcome.extract.dropped["zero_findings"] == 1
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
    assert report.dropped.get("no_pairs") == 1
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
    assert report.dropped["singleton_key"] == 1


def test_critic_stub_reports_not_implemented_and_points_at_the_adr():
    """Not a placeholder result. A stub that returned any ContradictionFinding at all would
    be worse than an empty list, because it manufactures a result where none exists."""
    report = critic_stub(n_clusters=4)
    assert report.status is StageStatus.not_implemented
    assert report.n_in == 4
    assert report.n_out == 0
    assert "ADR-0017" in (report.note or "")


def test_synthesis_stub_says_not_yet_built_rather_than_citing_a_decision():
    """Synthesis is unimplemented for a DIFFERENT reason than the Critic: no ADR retired
    it, it was simply never built. Dressing that up as a decision would misrepresent the
    project's own record."""
    report = synthesis_stub()
    assert report.status is StageStatus.not_implemented
    assert "ADR" not in (report.note or "")
