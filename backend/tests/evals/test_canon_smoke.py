import pytest

from biolit.config import get_settings


@pytest.mark.heavy
def test_real_ctd_dictionary_links_known_chemical():
    from biolit.canon.build_mesh import _download_tsv_gz
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary, build_alias_table

    s = get_settings()
    chem = _download_tsv_gz(s.ctd_chemicals_url)
    dis = _download_tsv_gz(s.ctd_diseases_url)
    linker = DictionaryLinker(MeshDictionary(build_alias_table(chem, dis)))
    result = linker.link("Metformin")
    assert result.concept is not None and result.concept.id == "MESH:D008687"


@pytest.mark.heavy
def test_real_bc5cdr_gold_parses_nonempty_with_mesh_ids():
    from biolit_evals.mesh_gold_download import load_bc5cdr_norm_gold

    gold = load_bc5cdr_norm_gold(get_settings().bc5cdr_cdr_zip_url)
    assert len(gold) > 1000  # BC5CDR test split has thousands of mentions
    assert any(g.mesh_ids for g in gold)  # at least some are linkable
