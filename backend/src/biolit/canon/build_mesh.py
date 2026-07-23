import csv
import gzip
import io
from pathlib import Path

import httpx

from biolit.canon.mesh import MeshDictionary, build_alias_table
from biolit.config import get_settings


def _download_tsv_gz(url: str) -> list[list[str]]:
    """Download a gzipped CTD TSV and return its data rows (comment lines dropped).

    Decoded as utf-8-sig: real CTD .tsv.gz downloads (like the fixtures used elsewhere
    in this project) can carry a leading UTF-8 BOM, which would otherwise survive into
    the first cell of the first row and defeat the `row[0].startswith("#")` comment
    filter below, leaking a header row into the alias table.
    """
    resp = httpx.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    return [row for row in reader if row and not row[0].startswith("#")]


def main() -> None:
    settings = get_settings()
    print("Downloading CTD chemicals + diseases ...")
    chem_rows = _download_tsv_gz(settings.ctd_chemicals_url)
    disease_rows = _download_tsv_gz(settings.ctd_diseases_url)
    table = build_alias_table(chem_rows, disease_rows)
    out = Path(settings.mesh_artifact_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    MeshDictionary(table).save_artifact(str(out))
    print(f"Wrote {len(table)} aliases to {out}")


if __name__ == "__main__":
    main()
