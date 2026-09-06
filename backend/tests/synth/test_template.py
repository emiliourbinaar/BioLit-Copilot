from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit.synth.template import ALREADY_CITED_MARKER, render_cluster


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


def test_render_cluster_quotes_nothing_twice_for_a_paper_already_cited():
    """A4. A paper legitimately belongs to several clusters -- it carries several
    chemical|disease pairs -- so membership must survive, but its finding sentences must not
    be re-quoted verbatim in every one. On the frozen metformin run 60 papers produced 129
    paper-appearances, so more than half the answer's quoted text is repeat."""
    cluster = Cluster(key="metformin|acidosis", paper_ids=["1"])
    papers = {"1": _paper("1", 2007, "N Engl J Med")}
    records = {"1": _record("1", "Lactate rose sharply.")}

    out = render_cluster(cluster, records, papers, already_cited={"1"})

    assert "PMID 1" in out, "membership is real and must still be shown"
    assert "Lactate rose sharply." not in out
    assert ALREADY_CITED_MARKER in out


def test_render_cluster_is_byte_identical_when_no_paper_was_cited_before():
    """Pins that A4 did not move the arm Gate A measured. `render_cluster`'s default is the
    exact function scored over 30 clusters (support 1.0, coverage 1.0, hallucinations 0), and
    dedup is a property of the JOIN across clusters, not of rendering one. If this ever
    fails, the committed Gate A numbers stop describing the shipped code."""
    cluster = Cluster(key="metformin|acidosis", paper_ids=["1", "2"])
    papers = {"1": _paper("1", 2007, "N Engl J Med"), "2": _paper("2", 2019, "Hum Reprod")}
    records = {"1": _record("1", "Lactate rose."), "2": _record("2", "It did not.")}

    assert render_cluster(cluster, records, papers) == render_cluster(
        cluster, records, papers, already_cited=set()
    )
    assert "Lactate rose." in render_cluster(cluster, records, papers)
