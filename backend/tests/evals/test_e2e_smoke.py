import pytest

from biolit.config import get_settings


@pytest.mark.heavy
def test_real_bc5cdr_documents_parse_and_offsets_align():
    from biolit_evals.mesh_gold_download import load_bc5cdr_documents

    docs = load_bc5cdr_documents(get_settings().bc5cdr_cdr_zip_url)
    assert len(docs) == 500  # BC5CDR test split
    total = 0
    for doc in docs:
        for m in doc.mentions:
            total += 1
            assert doc.text[m.start : m.end] == m.text, (doc.pmid, m)
    assert total == 9809  # matches the Phase 2 NER gold's tp+fn
