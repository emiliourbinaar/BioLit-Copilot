import gzip

import httpx
import respx

from biolit.canon.build_mesh import _download_tsv_gz


@respx.mock
def test_download_tsv_gz_strips_bom_so_comment_skip_works():
    # Real CTD .tsv.gz downloads can carry a leading UTF-8 BOM. If it survives into the
    # first cell of the first row, `row[0].startswith("#")` fails to recognize a comment
    # line and it leaks into the parsed rows as data.
    tsv = "# CTD comment header\nMetformin\tD008687\n"
    body = gzip.compress(tsv.encode("utf-8-sig"))
    respx.get("https://example.test/CTD_chemicals.tsv.gz").mock(
        return_value=httpx.Response(200, content=body)
    )

    rows = _download_tsv_gz("https://example.test/CTD_chemicals.tsv.gz")

    assert rows == [["Metformin", "D008687"]]
