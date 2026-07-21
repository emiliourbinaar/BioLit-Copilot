import biolit.ner  # noqa: F401
import biolit_evals  # noqa: F401  - package importable
from biolit.config import Settings


def test_ner_settings_defaults():
    settings = Settings()
    assert settings.ner_device == "auto"
    assert settings.ner_batch_size == 16
    assert settings.ner_score_threshold == 0.5
    assert "bc5cdr" in settings.ner_model_id.lower()
