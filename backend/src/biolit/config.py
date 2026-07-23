from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIOLIT_", env_file=".env", extra="ignore")

    ncbi_api_key: str | None = None
    ncbi_tool: str = "biolit-copilot"
    ncbi_email: str | None = None

    http_max_retries: int = 4
    http_backoff_base_seconds: float = 0.5
    http_backoff_max_seconds: float = 8.0

    ner_model_id: str = "Francesco-A/BiomedNLP-PubMedBERT-base-uncased-abstract-bc5cdr-ner-v1"
    ner_device: str = "auto"  # auto | cpu | cuda
    ner_cache_dir: str | None = None
    ner_batch_size: int = 16
    ner_score_threshold: float = 0.5

    mesh_artifact_path: str = "data/canon/mesh_aliases.json.gz"
    ctd_chemicals_url: str = "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"
    ctd_diseases_url: str = "https://ctdbase.org/reports/CTD_diseases.tsv.gz"
    bc5cdr_cdr_zip_url: str = (
        "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
