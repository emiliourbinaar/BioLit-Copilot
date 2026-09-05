from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord, Finding
from biolit.synth.template import render_cluster
from biolit_evals.synth_metrics import (
    MAX_ALIAS_WORDS,
    build_source_view,
    compression,
    coverage,
    dcr,
    hallucinated_concepts,
    judgment_language,
    judgment_volunteered,
    load_aliases,
    numerals,
    score_output,
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


def _paper_yj(pid: str, year: int, journal: str) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=None,
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


def test_journal_uniqueness_is_decided_on_the_normalised_journal_name():
    """The uniqueness guard and the match must agree on what "same journal" means. Keying
    uniqueness on the raw field while matching on tokens lets "N Engl J Med" and
    "N. Engl. J. Med." count as two distinct journals that both match one mention -- the
    bulk-credit hole the guard exists to close, reopened by the same normalisation drift
    Ruling 6 records."""
    cluster, records, papers = _fixture(p1="Alpha.", p2="Beta.")
    papers["p1"] = papers["p1"].model_copy(update={"year": 2019, "journal": "N Engl J Med"})
    papers["p2"] = papers["p2"].model_copy(update={"year": 2019, "journal": "N. Engl. J. Med."})

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


def test_dcr_counts_a_paper_retained_when_one_distinguishing_token_survives():
    cluster, records, papers = _fixture(
        p1="Metformin reduced hirsutism scores.",
        p2="Metformin reduced ovulation latency.",
    )

    got = dcr("Metformin reduced hirsutism in one report.", cluster, records)

    assert got.retained == 1
    assert got.lost == ("p2",)
    assert got.scorable == 2


def test_a_paper_with_no_distinguishing_tokens_is_excluded_from_the_denominator():
    """Two papers with identical findings cannot be told apart by any output, so scoring
    an arm on them would penalise it for the corpus rather than for its own behaviour."""
    cluster, records, papers = _fixture(p1="Identical finding.", p2="Identical finding.")

    got = dcr("Nothing in particular.", cluster, records)

    assert got.scorable == 0
    assert got.indistinguishable == ("p1", "p2")
    assert got.rate == 1.0


def test_a_distinguishing_token_inside_a_longer_word_is_not_retained():
    """Ruling 9. 'cyst' is a distinguishing token in p1's finding, and the output's
    'polycystic' contains it as a bare substring -- a substring test would count 'cyst'
    retained even though the word itself never occurs; token matching must not be fooled."""
    cluster, records, papers = _fixture(
        p1="Ovarian cyst volume decreased.",
        p2="Endometrial thickness increased.",
    )

    got = dcr("Polycystic ovary morphology was unchanged.", cluster, records)

    assert got.lost == ("p1", "p2")
    assert got.retained == 0


def test_a_paper_with_no_findings_is_separated_from_indistinguishable():
    """Ruling 10. A paper extraction produced nothing for means EXTRACTION failed; a paper
    with findings identical to a sibling's means the CORPUS carries a duplicate. Conflating
    the two would misreport which one happened, so they must land in different fields."""
    records = {"p1": _record("p1", "Metformin reduced hirsutism scores."), "p2": _record("p2")}
    cluster = Cluster(key="a|b", paper_ids=["p1", "p2"])

    got = dcr("Nothing about either paper.", cluster, records)

    assert got.no_findings == ("p2",)
    assert got.indistinguishable == ()
    assert got.scorable == 1


def test_judgment_language_is_detected_so_it_can_be_logged():
    """ADR-0018 established the pipeline cannot support an agreement claim. An LLM asked to
    characterise a cluster will volunteer one unprompted, so the rate is logged as a
    compliance diagnostic -- and deliberately never scored (spec §5.1)."""
    assert judgment_language("The findings are conflicting.") == ("conflicting",)
    assert judgment_language("Six papers measured HbA1c.") == ()


def test_judgment_language_detects_an_inflected_term_the_original_list_missed():
    """Ruling 14. Exact-token matching alone misses inflections of a stem already in the
    vocabulary -- measured before this extension, "contradicts" was one of four plainly
    judgmental sentences out of seven that produced no hit at all."""
    got = judgment_language("The 2019 study contradicts the 2007 finding.")

    assert got == ("contradicts",)


def test_judgment_language_splits_on_a_hyphen_unlike_the_mesh_tokeniser():
    """Ruling 15/23. `_alias_words` keeps '-' as a word character -- correct for MeSH
    ("non-hodgkin" must stay one token) and wrong here: "broadly-consistent" tokenised that
    way is one word, `broadly-consistent`, which is in no vocabulary and never fires.
    Judgment detection needs its own split so the hyphen is a word boundary."""
    got = judgment_language("The effect sizes were broadly-consistent across trials.")

    assert got == ("consistent",)


def test_judgment_language_detects_stem_inflections_the_exact_token_list_missed():
    """Ruling 23. `JUDGMENT_TERMS` grew to 39 exact tokens under Ruling 14 and still missed
    these, verified directly against the built module before this fix: "corroborated" and
    "refuting" are inflections of stems already in the list, "consistency" is absent while
    "inconsistency" is present, and the whole "concur" family is missing. A hand-maintained
    inflection list keeps losing this game -- stem-prefix matching makes the docstring's
    coverage claim true by construction instead of by vigilance."""
    assert judgment_language("The 2015 study corroborated the 2010 finding.") == ("corroborated",)
    assert judgment_language("The earlier paper is refuting this claim.") == ("refuting",)
    assert judgment_language("There is consistency across the trials.") == ("consistency",)
    assert judgment_language("The two studies concur on the mechanism.") == ("concur",)


def test_judgment_language_stems_do_not_fire_on_their_named_false_positives():
    """Ruling 23. The stem choice is deliberately narrower than the obvious one, and these
    are exactly the sentences that motivate it: "consisten", not "consist", so a cohort that
    "consists of" patients does not fire; and "concur" carries an explicit exclusion for
    "concurrent"/"concurrently", which mean "at the same time" and are ordinary biomedical
    vocabulary, not an agreement claim."""
    assert judgment_language("The cohort consists of 40 patients.") == ()
    assert judgment_language("Patients received concurrent chemotherapy.") == ()
    assert judgment_language("The trial ran concurrently at three sites.") == ()


def test_compression_is_measured_against_the_findings_not_the_metadata():
    """Spec §2 defines compression as "output length / concatenated source-findings length".
    Dividing by the whole source view instead folds the cluster key, journal, year and PMID of
    every paper into the denominator, so a cluster in a long-named journal would score as more
    compressed than the same findings in a short-named one -- and "below 1.0 means the arm said
    it shorter" would stop being true. Metadata length must not move this number."""
    short = _fixture(p1="Metformin reduced hirsutism.", p2="Ovulation rose.")
    long_journal = _fixture(p1="Metformin reduced hirsutism.", p2="Ovulation rose.")
    for pid in ("p1", "p2"):
        long_journal[2][pid] = long_journal[2][pid].model_copy(
            update={"journal": "Journal of Deliberately Very Long Titles and Subtitles"}
        )

    output = "Two papers."
    a = compression(output, build_source_view(*short))
    b = compression(output, build_source_view(*long_journal))

    assert a == b


def test_score_output_runs_every_metric_over_one_output():
    """The brief asserted `compression <= 1.5`, and the plan's Ruling 2 predicted "slightly
    above 1.0". Both were wrong, and an upper bound was the wrong shape anyway: the number it
    pins depends on how long the fixture's findings happen to be (measured 2.96 on this
    two-sentence fixture, 1.28 on realistic abstract-length ones), so any constant here is a
    magic number waiting to break. `> 1.0` is instead a structural invariant of the control:
    the template reproduces every finding verbatim and then adds a heading, a count line, a
    stamp per paper and quoting, so it can never be shorter than the findings alone. If this
    ever fails, the template has started dropping content."""
    cluster, records, papers = _fixture(p1="Metformin reduced hirsutism.", p2="Ovulation rose.")
    out = render_cluster(cluster, records, papers)

    score = score_output(out, cluster, records, papers, aliases={})

    assert score.coverage.rate == 1.0
    assert score.support.rate == 1.0
    assert score.compression > 1.0
    assert score.hallucinated == ()
    assert score.dcr == dcr(out, cluster, records)

    # judgment_terms must be wired from `output`, not `source.text` -- and the template's
    # own rendering can never distinguish the two, because it invents no prose beyond the
    # source it was given, so this half of the wiring needs an output that actually
    # diverges from the source: judgment language the source does not contain.
    judgy_output = out + " The findings are conflicting."
    judgy_score = score_output(judgy_output, cluster, records, papers, aliases={})
    assert judgy_score.judgment_terms == judgment_language(judgy_output)

    # Ruling 27. The control arm volunteers nothing -- it quotes. That is the construction-true
    # self-test the plan wanted from `judgment_terms` and never had.
    source = build_source_view(cluster, records, papers)
    assert score.judgment_volunteered == ()
    assert judgy_score.judgment_volunteered == judgment_volunteered(judgy_output, source)
    assert "conflicting" in judgy_score.judgment_volunteered


def test_a_greek_suffixed_entity_the_source_named_is_not_counted_as_hallucinated():
    """Ruling 28. `_WORD_SPLIT` treats non-ASCII as a separator, so "TNF-α" tokenised to
    "tnf-" with the hyphen left dangling, and "tnf-" can never match the alias key "tnf".
    The source therefore appeared not to mention TNF at all, and an arm writing the plain
    form was reported as inventing it -- a false positive on the ONE hard disqualifier that
    must be exactly 0. Measured on the frozen corpus: 9 such tokens, including tnf-, tgf-
    and nf-, three of the most common entities in this literature.

    ⚠️ The Greek exemplar this test originally used is SUPERSEDED by Ruling 29 and moved to
    its own test. Folding resolves "TNF-α" to `tnf-alpha` -> D014409, the correct specific
    concept, rather than crushing it to the broader `tnf` -> D014923 as the strip alone did.
    The strip is still load-bearing for hyphens orphaned by anything folding does not repair
    -- six such tokens survive on the real corpus, among them "-induced" and "depressive-" --
    and a trailing hyphen from PDF line-break hyphenation is the plainest instance.
    """
    cluster, records, papers = _fixture(
        p1="Serum TNF- rose sharply after treatment.",
        p2="Ovulation rose.",
    )
    source = build_source_view(cluster, records, papers)

    assert hallucinated_concepts("TNF was elevated.", source, {"tnf": "D014923"}) == ()


def test_coverage_does_not_credit_a_paper_from_a_year_and_journal_borrowed_from_others():
    """Ruling 31. Ruling 12 closed BULK credit, where several papers share one (year,
    journal) pair. It did not close the CROSS-PRODUCT: the year and the journal were checked
    independently, so a year donated by one paper and a journal donated by another credited a
    third that the output never mentioned. All three pairs below are unique, so Ruling 12's
    guard does not engage at all.

    Live on the frozen sample this let 5 papers be omitted entirely while coverage still read
    1.000, one of them converting a true 0.889 -- BELOW the 0.90 disqualifier -- into a pass.
    A disqualifier that cannot fire, which is the shape this harness exists to catch, and
    template-invisible because the template cites every PMID.
    """
    papers = {
        "1": _paper_yj("1", 2019, "Lancet"),
        "2": _paper_yj("2", 2019, "N Engl J Med"),
        "3": _paper_yj("3", 2021, "Lancet"),
    }
    cluster = Cluster(key="a|b", paper_ids=["1", "2", "3"])

    got = coverage("The 2019 N Engl J Med report and the 2021 Lancet report.", cluster, papers)

    assert got.missing == ("1",)


def test_an_ascii_rendering_of_a_greek_lettered_entity_is_not_hallucinated():
    """Ruling 29. Ruling 28 stripped DANGLING hyphens, which fixed a Greek letter in final
    position ("TNF-α") and closed the entry -- but the CLASS stayed open for a non-ASCII
    character in INTERIOR position. "TGF-β1" tokenised to `tgf` with the `1` orphaned, so the
    source read as never naming D053773, and an arm writing `TGF-beta1` -- the standard ASCII
    rendering, and what a plain-text model routinely emits -- was reported as inventing it.
    Hard disqualifier, must be exactly 0, and template-invisible: the control quotes verbatim
    so its surface forms always match, and `hallucinated_total` stayed 0 either way.
    """
    cluster, records, papers = _fixture(
        p1="Serum TGF-β1 rose after treatment.",
        p2="Ovulation rose.",
    )
    source = build_source_view(cluster, records, papers)
    aliases = {"tgf-beta1": "D053773"}

    assert hallucinated_concepts("TGF-beta1 was elevated.", source, aliases) == ()
    assert hallucinated_concepts("TGF-β1 was elevated.", source, aliases) == ()


def test_judgment_volunteered_excludes_vocabulary_the_source_itself_used():
    """Ruling 27, measured on the frozen sample: the template scored 10/30 on
    `judgment_terms` where the plan asserted 0 "by construction". All 12 hits were verbatim
    quoted source sentences using the words CLINICALLY -- "consistent with exertional
    rhabdomyolysis" is a diagnostic impression, not a claim about another paper. The template
    quotes, so it inherits the corpus's vocabulary; 0 was never construction-true.

    `judgment_volunteered` is the check the plan was reaching for and IS construction-true
    for the template: its output is boilerplate plus verbatim source, and none of its
    boilerplate is judgment vocabulary. Both numbers are reported because neither suffices
    alone -- subtracting the source hides an arm that writes "the findings are consistent
    across studies" over a corpus that says "consistent with rhabdomyolysis", and counting
    everything drowns the signal in clinical prose.
    """
    cluster, records, papers = _fixture(
        p1="Elevated creatine kinase, consistent with rhabdomyolysis.",
        p2="Ovulation rose.",
    )
    source = build_source_view(cluster, records, papers)

    quoted = 'The report noted findings "consistent with rhabdomyolysis".'
    assert judgment_language(quoted) == ("consistent",)
    assert judgment_volunteered(quoted, source) == ()

    volunteered = "The two papers contradict each other."
    assert judgment_volunteered(volunteered, source) == ("contradict",)

    # Ruling 30. The laundering path: the arm makes an epistemic claim using a word the
    # corpus happens to use CLINICALLY. Bare membership cannot see it -- "consistent" is in
    # the source, so subtraction returns () and the headline reads 0. Measured at 10/30
    # clusters on the frozen sample, i.e. a third of the gate's sample.
    laundered = "The findings are consistent across these studies."
    assert judgment_language(laundered) == ("consistent",)
    assert judgment_volunteered(laundered, source) == ("consistent",)


def test_dcr_with_a_lower_min_token_len_retains_a_short_marker_the_default_floor_drops():
    """Ruling 16. IL6/TNF are realistic 3-character biomedical markers, shorter than the
    default floor of 4. The default excludes them from distinguishing_tokens entirely, so a
    paper identified only by its marker is lost; min_token_len=3 keeps them and retains it."""
    cluster, records, papers = _fixture(
        p1="IL6 levels increased with treatment.",
        p2="TNF levels decreased with treatment.",
    )
    output = "The IL6 finding was notable."

    default = dcr(output, cluster, records)
    lowered = dcr(output, cluster, records, min_token_len=3)

    assert "p1" in default.lost
    assert "p1" not in lowered.lost
    assert lowered.retained == 1
