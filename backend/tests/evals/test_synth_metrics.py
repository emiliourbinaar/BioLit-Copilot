from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit_evals.synth_metrics import build_source_view, numerals, support_rate


def _paper(pid: str, year: int = 2010) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        text_type=TextType.abstract_only,
        year=year,
        journal="J Test",
    )


def _record(pid: str, *sentences: str) -> ExtractedRecord:
    return ExtractedRecord(
        paper_id=pid,
        key_findings=[
            Finding(text=text, start=0, end=len(text), sentence_index=i)
            for i, text in enumerate(sentences)
        ],
    )


def _fixture(**findings: str):
    cluster = Cluster(key="a|b", paper_ids=sorted(findings))
    papers = {pid: _paper(pid) for pid in findings}
    records = {pid: _record(pid, text) for pid, text in findings.items()}
    return cluster, records, papers


def test_numerals_finds_integers_decimals_and_percentages():
    assert numerals("HbA1c fell 1.5% in 42 of 100 patients") == ["1.5", "42", "100"]


def test_support_rate_counts_a_numeral_absent_from_source_as_unsupported():
    cluster, records, papers = _fixture(p1="HbA1c fell 1.5% over 12 weeks.")
    source = build_source_view(cluster, records, papers)

    got = support_rate("HbA1c fell 1.5% over 24 weeks.", source)

    assert got.unsupported == ("24",)
    assert got.supported == 1
    assert got.total == 2


def test_the_cluster_paper_count_is_a_supported_numeral():
    """The one aggregate an arm may legitimately introduce. Without this exemption the
    metric would penalise 'three papers report...' on a three-paper cluster."""
    cluster, records, papers = _fixture(p1="Alpha.", p2="Beta.", p3="Gamma.")
    source = build_source_view(cluster, records, papers)

    got = support_rate("3 papers discuss this pair.", source)

    assert got.unsupported == ()
    assert got.rate == 1.0
