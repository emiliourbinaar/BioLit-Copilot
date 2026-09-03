from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit_evals.synth_metrics import (
    MAX_ALIAS_WORDS,
    build_source_view,
    coverage,
    hallucinated_concepts,
    load_aliases,
    numerals,
    support_rate,
)


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


def test_a_numeral_written_flush_against_its_unit_is_still_a_numeral():
    """Dose and duration are routinely written without a space. A trailing word boundary
    drops "500mg" entirely -- letting a fabricated dose evade the support disqualifier --
    and truncates "1.5mg/kg" to a phantom "1" that was never in the text at all."""
    assert numerals("metformin 500mg twice daily for 12weeks") == ["500", "12"]
    assert numerals("1.5mg/kg") == ["1.5"]


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


def test_an_entity_absent_from_the_source_is_reported_as_hallucinated():
    """The dangerous failure: inventing a drug or disease that no source paper mentions."""
    cluster, records, papers = _fixture(p1="Metformin lowered glucose.")
    source = build_source_view(cluster, records, papers)
    aliases = {"metformin": "D008687", "rosiglitazone": "D000077154"}

    got = hallucinated_concepts("Metformin and rosiglitazone lowered glucose.", source, aliases)

    assert got == ("D000077154",)


def test_a_multi_word_alias_is_matched():
    cluster, records, papers = _fixture(p1="Nothing relevant here.")
    source = build_source_view(cluster, records, papers)
    aliases = {"polycystic ovary syndrome": "D011085"}

    got = hallucinated_concepts(
        "Patients with polycystic ovary syndrome improved.", source, aliases
    )

    assert got == ("D011085",)


def test_a_punctuated_alias_is_matched_after_loading(tmp_path):
    """The MeSH artifact stores a concept's own inverted name as its primary alias, e.g.
    "Diabetes Mellitus, Type 2" -- comma and all. `load_aliases` used to key on that string
    merely lowercased, while the scanner's n-gram keys never contain punctuation at all, so
    a punctuated alias like this one could never be matched from either side."""
    import gzip
    import json

    path = tmp_path / "aliases.json.gz"
    payload = {
        "Diabetes Mellitus, Type 2": [
            ["MESH:D003924", "Diabetes Mellitus, Type 2", True],
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    aliases = load_aliases(str(path))

    cluster, records, papers = _fixture(p1="Nothing relevant here.")
    source = build_source_view(cluster, records, papers)

    got = hallucinated_concepts(
        "The cohort included patients with diabetes mellitus type 2.", source, aliases
    )

    assert got == ("D003924",)


def test_an_alias_longer_than_the_word_cap_is_not_matched():
    """MAX_ALIAS_WORDS caps the n-gram scan at 6 words -- a deliberate, documented limit that
    trades a small amount of coverage for keeping the scan linear in text length. An alias
    one word past the cap is therefore invisible to the scanner even when present verbatim."""
    cluster, records, papers = _fixture(p1="Nothing relevant here.")
    source = build_source_view(cluster, records, papers)
    words = [f"word{i}" for i in range(MAX_ALIAS_WORDS + 1)]
    alias = " ".join(words)
    aliases = {alias: "D000001"}

    got = hallucinated_concepts(alias, source, aliases)

    assert got == ()


def test_coverage_reports_the_papers_the_output_never_references():
    cluster, records, papers = _fixture(p1="Alpha.", p2="Beta.", p3="Gamma.")

    got = coverage("Discussion of PMID p1 and PMID p3 only.", cluster, papers)

    assert got.missing == ("p2",)
    assert got.covered == 2
    assert got.n_papers == 3


def test_coverage_does_not_credit_a_pid_that_is_only_a_substring_of_another():
    """'1234567' in 'PMID 12345678' is True, so a naive substring test would score the
    shorter PMID covered by an output that only ever named the longer one. That is a false
    "covered" on a disqualifier -- a disqualifier that cannot fire."""
    papers = {
        "1234567": _paper("1234567"),
        "12345678": _paper("12345678"),
    }
    cluster = Cluster(key="a|b", paper_ids=sorted(papers))

    got = coverage("Full data reported in PMID 12345678.", cluster, papers)

    assert got.missing == ("1234567",)


def test_a_doi_shaped_paper_id_with_no_pmid_cited_verbatim_is_covered():
    """`Paper.id` is `doi or pmid`, and bioRxiv's client sets `id=doi or title` with no
    `pmid` at all. A DOI or title id routinely contains punctuation that `_alias_words`
    splits into several tokens, so a single-token equality check could never credit it --
    even when the output cites it verbatim."""
    doi = "10.1234/synth.gate.2019"
    paper = Paper(
        id=doi,
        source=Source.biorxiv,
        pmid=None,
        title="A synthetic preprint",
        text_type=TextType.abstract_only,
        year=2019,
        journal=None,
    )
    cluster = Cluster(key="a|b", paper_ids=[doi])

    got = coverage(f"Findings are reported in {doi}.", cluster, {doi: paper})

    assert got.missing == ()


def test_a_paper_identified_only_by_year_and_journal_is_covered():
    """Spec §2's paper reference is PMID *or* year+journal. Spec §5 hands each arm the year
    and journal, so "the 2019 N Engl J Med study" is an identification the spec allows --
    with no PMID anywhere in the output."""
    cluster, records, papers = _fixture(p1="Metformin lowered glucose.")
    papers["p1"] = papers["p1"].model_copy(update={"year": 2019, "journal": "N Engl J Med"})

    output = "The 2019 N Engl J Med study found metformin lowered glucose."
    got = coverage(output, cluster, papers)

    assert got.missing == ()


def test_two_papers_sharing_year_and_journal_are_not_covered_by_one_mention():
    """A (year, journal) pair that two papers share cannot distinguish between them, so a
    single mention such as "the 2019 NEJM papers" must not cover both -- a disqualifier
    satisfiable in bulk is a disqualifier that cannot fire."""
    cluster, records, papers = _fixture(p1="Alpha.", p2="Beta.")
    papers["p1"] = papers["p1"].model_copy(update={"year": 2019, "journal": "N Engl J Med"})
    papers["p2"] = papers["p2"].model_copy(update={"year": 2019, "journal": "N Engl J Med"})

    got = coverage("The 2019 N Engl J Med papers found similar effects.", cluster, papers)

    assert got.missing == ("p1", "p2")


def test_a_shorter_pmid_that_is_a_substring_of_a_longer_one_is_still_reported_missing():
    """Ruling 8, re-verified under phrase matching. `'1234567' in 'PMID 12345678'` is true
    as a substring, but `['1234567']` is not a contiguous run of `['pmid', '12345678']` --
    phrase matching must still refuse to credit the shorter PMID."""
    papers = {
        "1234567": _paper("1234567"),
        "12345678": _paper("12345678"),
    }
    cluster = Cluster(key="a|b", paper_ids=sorted(papers))

    got = coverage("Full data reported for PMID 12345678.", cluster, papers)

    assert got.missing == ("1234567",)
