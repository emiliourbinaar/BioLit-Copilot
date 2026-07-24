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


@pytest.mark.heavy
def test_real_long_abstract_is_processed_beyond_the_512_token_cutoff():
    """Regression for the overflow this eval surfaced.

    Before windowing, any document over 512 tokens raised
    `RuntimeError: The size of tensor a (549) must match the size of tensor b (512)`.
    2.2% of real BC5CDR abstracts exceed it. Asserting "no crash" alone would pass even if
    the tail were silently truncated, so this also requires entities recovered from text
    beyond the old cutoff.
    """
    from transformers import AutoTokenizer

    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit_evals.mesh_gold_download import load_bc5cdr_documents

    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.ner_model_id)
    documents = load_bc5cdr_documents(settings.bc5cdr_cdr_zip_url)

    long_docs = [d for d in documents if len(tokenizer(d.text)["input_ids"]) > 512]
    assert long_docs, "expected some BC5CDR abstracts to exceed the 512-token limit"

    doc = max(long_docs, key=lambda d: len(d.text))
    entities = extract_entities(doc.text, NerModel.load(settings))
    assert entities, "no entities extracted from a long abstract"

    # Character offset where the old 512-token limit would have cut the document.
    truncated_at = len(tokenizer.decode(tokenizer(doc.text)["input_ids"][:512]))
    beyond = [e for e in entities if e.start is not None and e.start > truncated_at]
    assert beyond, f"no entities found past char {truncated_at}; the tail is still being dropped"
    for entity in entities:
        assert doc.text[entity.start : entity.end] == entity.text
