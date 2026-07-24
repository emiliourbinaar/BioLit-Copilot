import io
import zipfile

import httpx

from biolit_evals.mesh_gold import (
    GoldDocument,
    GoldMention,
    parse_pubtator,
    parse_pubtator_documents,
)

# PubTator file inside CDR_Data.zip (BioCreative V CDR corpus, test split).
_TEST_MEMBER = "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt"


def load_bc5cdr_norm_gold(zip_url: str, member: str = _TEST_MEMBER) -> list[GoldMention]:
    """Download CDR_Data.zip (ungated bigbio/bc5cdr mirror) and parse its test-split
    PubTator file into gold mentions carrying MeSH IDs. Heavy/manual (network + ~20 MB)."""
    resp = httpx.get(zip_url, follow_redirects=True, timeout=300.0)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pubtator = zf.read(member).decode("utf-8")
    return parse_pubtator(pubtator)


def load_bc5cdr_documents(zip_url: str, member: str = _TEST_MEMBER) -> list[GoldDocument]:
    """Download CDR_Data.zip and parse its test split into documents with their own text.

    Heavy/manual (network + ~20 MB), same source and member as `load_bc5cdr_norm_gold`.
    """
    resp = httpx.get(zip_url, follow_redirects=True, timeout=300.0)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        pubtator = zf.read(member).decode("utf-8")
    return parse_pubtator_documents(pubtator)
