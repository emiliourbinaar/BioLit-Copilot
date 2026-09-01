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
        <article-id pub-id-type="pmc">3333333</article-id>
        <permissions>
          <license>
            <license-p>This file is available for text mining.</license-p>
          </license>
        </permissions>
      </article-meta></front></article></pmc-articleset>"""
    assert licences_by_pmcid(ET.fromstring(xml)) == {"3333333": None}
