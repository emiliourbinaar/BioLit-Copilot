from biolit import __version__
from biolit.config import get_settings


def test_version_present():
    assert __version__ == "0.1.0"


def test_settings_defaults():
    settings = get_settings()
    assert settings.http_max_retries == 4
    assert settings.ncbi_tool == "biolit-copilot"
