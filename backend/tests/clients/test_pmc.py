from pathlib import Path
from xml.etree import ElementTree as ET

from biolit.clients.pmc import licences_by_pmcid, normalize_pmcid

CASSETTES = Path(__file__).parent.parent / "cassettes"


def _root(name: str) -> ET.Element:
    return ET.fromstring((CASSETTES / name).read_text(encoding="utf-8"))


def test_normalize_pmcid_strips_the_prefix_so_both_sources_key_alike():
    """PubMed's ArticleIdList reports `PMC8917620`; PMC's own article-id reports the bare
    digits `8917620`. Keying the two together without normalising yields an empty join and
    therefore zero permitted papers -- silently, and indistinguishably from 'nothing is
    licensed'. Both directions and the None case are pinned."""
    assert normalize_pmcid("PMC8917620") == "8917620"
    assert normalize_pmcid("8917620") == "8917620"
    assert normalize_pmcid("pmc8917620") == "8917620"
    assert normalize_pmcid(None) is None
    assert normalize_pmcid("") is None


def test_licence_is_read_from_the_ali_license_ref():
    licences = licences_by_pmcid(_root("pmc_efetch_cc_by.xml"))
    assert licences["8917620"] == "https://creativecommons.org/licenses/by/4.0/"


def test_licence_falls_back_to_the_license_href_when_there_is_no_ali_ref():
    """Not every publisher emits ali:license_ref. The fallback is the `<license>` element's
    xlink:href -- still an identifier, never the element's prose."""
    licences = licences_by_pmcid(_root("pmc_efetch_cc_by.xml"))
    assert licences["7654321"] == "https://creativecommons.org/licenses/by-nc-nd/4.0/"


def test_a_restricted_stub_yields_no_licence_rather_than_a_guess():
    """The real PMC1401093 shape -- the article that 404'd throughout Phase 5. It comes back
    as a ~6KB stub with no <permissions> and no <body>. It must map to None, which refuses.

    Classification keys on the permissions block ALONE. <body> presence is corroborating
    evidence that these are different documents, not a criterion: an article could carry a
    permissive licence with no body shipped, and refusing it on that basis would be wrong.
    """
    licences = licences_by_pmcid(_root("pmc_efetch_restricted_stub.xml"))
    assert licences == {"1401093": None}


def test_prose_is_never_returned_as_a_licence():
    """The NIH 'available for text mining' statement lives in <license-p> prose with no
    identifier. Returning that prose would push a non-licence into the token normalizer and
    invite a permissive match on wording. Only identifiers are ever returned."""
    xml = """<pmc-articleset><article
        xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
        xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-id pub-id-type="pmcid">PMC3333333</article-id>
        <permissions>
          <license>
            <license-p>This file is available for text mining.</license-p>
          </license>
        </permissions>
      </article-meta></front></article></pmc-articleset>"""
    assert licences_by_pmcid(ET.fromstring(xml)) == {"3333333": None}


def test_only_the_emitted_pmcid_type_is_accepted_so_fixtures_cannot_drift_from_the_service():
    """The cassettes must reproduce what efetch db=pmc actually emits, not a convenient shape.

    They originally used `pub-id-type="pmc"` with bare digits. The live service emits
    `pmcid` with the PMC-prefixed value, alongside `pmcid-ver`, `pmcaid` and `pmcaiid` --
    verified across 10 articles spanning the id range, none of which emitted a bare `pmc`.
    Against the fictional fixtures the parser passed while being unable to read a single
    real response; the fixtures were the defect, so they were corrected and the parser
    narrowed to the emitted type rather than widened to accept both.

    Mutation-verified in both directions: the pre-correction parser (`== "pmc"`) fails four
    of the five tests in this file against the corrected fixtures and passed all five against
    the old ones, and re-broadening to `in ("pmc", "pmcid")` fails this test.

    `pmcid-ver` carries a versioned value (`PMC8917620.1`) for the SAME article, so the match
    is exact rather than a prefix test -- otherwise one article could yield two keys.
    """
    xml = """<pmc-articleset><article>
      <front><article-meta>
        <article-id pub-id-type="pmc">9999999</article-id>
      </article-meta></front></article></pmc-articleset>"""
    assert licences_by_pmcid(ET.fromstring(xml)) == {}

    versioned = """<pmc-articleset><article>
      <front><article-meta>
        <article-id pub-id-type="pmcid">PMC8917620</article-id>
        <article-id pub-id-type="pmcid-ver">PMC8917620.1</article-id>
      </article-meta></front></article></pmc-articleset>"""
    assert licences_by_pmcid(ET.fromstring(versioned)) == {"8917620": None}
