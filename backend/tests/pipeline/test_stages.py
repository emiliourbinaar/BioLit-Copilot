from biolit.domain.enums import EntityLabel, LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.pipeline.stages import (
    entities_stage,
    records_stage,
    retrieve_stage,
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
