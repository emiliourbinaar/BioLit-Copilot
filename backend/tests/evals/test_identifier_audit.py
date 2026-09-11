from biolit_evals.identifier_audit import audit_state, paper_verdict


def _stored(pmid: str, *, doi: str | None, pmc: str | None, allowed: bool) -> dict:
    """A paper as a frozen `--json-out` state recorded it: `raw.pmc_id` is the id the licence
    lookup ACTUALLY used, which is the evidence this audit rests on."""
    return {
        "id": doi or pmid,
        "pmid": pmid,
        "doi": doi,
        "extraction_allowed": allowed,
        "raw": {"pmid": pmid, "pmc_id": pmc},
    }


def test_a_paper_licensed_under_a_pmc_id_that_is_not_its_own_is_wrongly_allowed():
    """DEF-0008's two symptoms, judged against the paper's OWN identifiers from PubMed.

    A licence looked up under someone else's PMC id says nothing about this paper, so an
    allowed paper whose lookup id is not its own was allowed wrongly -- regardless of what that
    other licence said. A refused paper cannot have been wrongly ALLOWED, whatever its ids.
    """
    borrowed = paper_verdict(
        _stored("2", doi="10.1/cited", pmc="1111111", allowed=True),
        own_doi="10.1/own",
        own_pmc=None,
    )
    assert borrowed.wrongly_allowed and not borrowed.doi_is_own and not borrowed.licence_is_own

    honest = paper_verdict(
        _stored("1", doi="10.1/a", pmc="7", allowed=True), own_doi="10.1/a", own_pmc="PMC7"
    )
    assert not honest.wrongly_allowed and honest.doi_is_own and honest.licence_is_own

    refused = paper_verdict(
        _stored("3", doi="10.1/x", pmc="9", allowed=False), own_doi="10.1/y", own_pmc=None
    )
    assert not refused.wrongly_allowed and not refused.doi_is_own


def test_a_state_audit_traces_each_wrongly_allowed_paper_into_everything_built_on_it():
    """A borrowed licence matters by what it let through. The audit must say which records,
    which clusters and which quoted answer lines rest on a wrongly-allowed paper -- and must
    count stored `Paper.id` collisions, since DEF-0008 manufactured DEF-0007's.
    """
    state = {
        "question": "q",
        "candidate_papers": [
            _stored("1", doi="10.1/a", pmc="7", allowed=True),
            _stored("2", doi="10.1/cited", pmc="1111111", allowed=True),
            _stored("3", doi="10.1/cited", pmc=None, allowed=False),
        ],
        "extracted_records": {"10.1/a": {}, "10.1/cited": {}},
        "clusters": [
            {"key": "A|B", "paper_ids": ["10.1/a", "10.1/cited"]},
            {"key": "C|D", "paper_ids": ["10.1/a"]},
        ],
        "answer": '- 2025 · J · PMID 2\n  "a quote"\n- 2024 · J · PMID 1\n',
    }
    own = {"1": ("10.1/a", "PMC7"), "2": ("10.1/own-2", None), "3": ("10.1/own-3", None)}

    audit = audit_state(state, own)

    assert audit.wrongly_allowed_pmids == ["2"]
    assert audit.n_doi_not_own == 2
    assert audit.stored_id_collisions == [["2", "3"]]
    assert audit.records_on_wrong == ["10.1/cited"]
    assert audit.clusters_on_wrong == [("A|B", 2, 1)]
    assert audit.answer_cites_wrong == ["2"]
    assert audit.answer_present


def test_what_the_audit_cannot_see_is_reported_as_unseen_never_as_clean():
    """Two false zeros, both met on the real corpora. A PMID PubMed no longer returns cannot be
    judged, so it is listed as unverifiable rather than passed; and the frozen states store NO
    answer text (Gate A rendered answers later), so "cites 0 wrongly-allowed papers" there
    would mean "had nothing to cite", which must not read as a clean result.
    """
    state = {
        "question": "q",
        "candidate_papers": [
            _stored("1", doi="10.1/a", pmc="7", allowed=True),
            _stored("9", doi="10.1/gone", pmc="8", allowed=True),
        ],
        "extracted_records": {},
        "clusters": [],
        "answer": "",
    }

    audit = audit_state(state, {"1": ("10.1/a", "PMC7")})

    assert audit.unverifiable_pmids == ["9"]
    assert audit.wrongly_allowed_pmids == []
    assert not audit.answer_present
