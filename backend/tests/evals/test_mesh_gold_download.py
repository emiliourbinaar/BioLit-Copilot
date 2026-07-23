import io
import zipfile

import httpx
import respx

from biolit_evals.mesh_gold_download import load_bc5cdr_norm_gold

_MEMBER = "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt"


def _zip_bytes(member_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(_MEMBER, member_text)
    return buf.getvalue()


@respx.mock
def test_load_bc5cdr_norm_gold_downloads_and_parses_test_member():
    pubtator = "1\t0\t9\tMetformin\tChemical\tD008687\n"
    respx.get("https://example.test/CDR_Data.zip").mock(
        return_value=httpx.Response(200, content=_zip_bytes(pubtator))
    )

    gold = load_bc5cdr_norm_gold("https://example.test/CDR_Data.zip")

    assert len(gold) == 1
    assert gold[0].pmid == "1"
    assert gold[0].mesh_ids == ("MESH:D008687",)
