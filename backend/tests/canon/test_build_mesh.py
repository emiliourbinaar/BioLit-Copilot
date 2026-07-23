import gzip

import httpx
import respx

from biolit.canon.build_mesh import _download_ctd


@respx.mock
def test_download_ctd_strips_bom_so_header_detection_works():
    # Real CTD .tsv.gz downloads can carry a leading UTF-8 BOM. If it survives, the
    # header line no longer starts with "#", so build_alias_table cannot find the
    # column header and would misread the dump.
    tsv = "# ChemicalName\tChemicalID\nMetformin\tMESH:D008687\n"
    body = gzip.compress(tsv.encode("utf-8-sig"))
    respx.get("https://example.test/CTD_chemicals.tsv.gz").mock(
        return_value=httpx.Response(200, content=body)
    )

    text = _download_ctd("https://example.test/CTD_chemicals.tsv.gz")

    assert text.startswith("# ChemicalName")  # BOM stripped
    assert text.splitlines()[1] == "Metformin\tMESH:D008687"
