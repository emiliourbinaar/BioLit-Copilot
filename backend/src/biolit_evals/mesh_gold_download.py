import io
import zipfile

import httpx

from biolit_evals.mesh_gold import GoldMention, parse_pubtator

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
