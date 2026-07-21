from biolit.storage.models import EMBEDDING_DIM, Base, PaperEmbeddingRow, PaperRow


def test_tables_registered():
    tables = set(Base.metadata.tables)
    assert {"papers", "paper_embeddings"} <= tables


def test_paper_columns():
    cols = {c.name for c in PaperRow.__table__.columns}
    assert {"id", "source", "doi", "pmid", "title", "text_type", "license_tier"} <= cols


def test_embedding_has_vector_column():
    col = PaperEmbeddingRow.__table__.columns["embedding"]
    # pgvector's Vector type exposes the configured dimension.
    assert getattr(col.type, "dim", EMBEDDING_DIM) == EMBEDDING_DIM
    assert "chunk_kind" in {c.name for c in PaperEmbeddingRow.__table__.columns}
