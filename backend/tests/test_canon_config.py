from biolit.config import Settings


def test_canon_settings_defaults():
    settings = Settings()
    assert settings.mesh_artifact_path == "data/canon/mesh_aliases.json.gz"
    assert settings.ctd_chemicals_url == "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"
    assert settings.ctd_diseases_url == "https://ctdbase.org/reports/CTD_diseases.tsv.gz"
    assert (
        settings.bc5cdr_cdr_zip_url
        == "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"
    )
