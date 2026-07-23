import gzip
from pathlib import Path

import httpx

from biolit.canon.mesh import MeshDictionary, build_alias_table
from biolit.config import get_settings


def _download_ctd(url: str) -> str:
    """Download a gzipped CTD TSV and return its decoded text.

    Decoded as utf-8-sig so a leading UTF-8 BOM (which real CTD dumps can carry) is
    stripped and cannot defeat the '#' header/comment detection in build_alias_table.
    """
    resp = httpx.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    return gzip.decompress(resp.content).decode("utf-8-sig")


def main() -> None:
    settings = get_settings()
    print("Downloading CTD chemicals + diseases ...")
    chem_text = _download_ctd(settings.ctd_chemicals_url)
    disease_text = _download_ctd(settings.ctd_diseases_url)
    table = build_alias_table(chem_text, disease_text)
    out = Path(settings.mesh_artifact_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    MeshDictionary(table).save_artifact(str(out))
    print(f"Wrote {len(table)} aliases to {out}")


if __name__ == "__main__":
    main()
