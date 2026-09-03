from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit.synth.template import render_cluster


def _paper(pid: str, year: int | None, journal: str | None) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        text_type=TextType.abstract_only,
        year=year,
        journal=journal,
    )


def _record(pid: str, *sentences: str) -> ExtractedRecord:
    return ExtractedRecord(
        paper_id=pid,
        key_findings=[
            Finding(text=text, start=0, end=len(text), sentence_index=i)
            for i, text in enumerate(sentences)
        ],
    )


def test_render_cluster_lists_papers_oldest_first():
    cluster = Cluster(key="metformin|PCOS", paper_ids=["2", "1"])
    papers = {"1": _paper("1", 2007, "N Engl J Med"), "2": _paper("2", 2019, "Hum Reprod")}
    records = {"1": _record("1", "Clomiphene beat metformin."), "2": _record("2", "OHSS fell.")}

    out = render_cluster(cluster, records, papers)

    assert out.index("PMID 1") < out.index("PMID 2")
    assert "metformin — PCOS" in out
    assert "2 papers, 2007–2019." in out


def test_a_paper_with_no_findings_is_marked_not_dropped():
    """`zero_findings` is a tracked ledger key. Omitting those papers would inflate this
    arm's coverage against the very metric coverage is meant to measure."""
    cluster = Cluster(key="a|b", paper_ids=["1", "2"])
    papers = {"1": _paper("1", 2001, "J One"), "2": _paper("2", 2002, "J Two")}
    records = {"1": _record("1", "A finding.")}

    out = render_cluster(cluster, records, papers)

    assert "PMID 2" in out
    assert "(no finding sentence extracted)" in out
