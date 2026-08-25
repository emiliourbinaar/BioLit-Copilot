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
    ctd_chemicals_diseases_url: str = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
    bc5cdr_cdr_zip_url: str = (
        "https://huggingface.co/datasets/bigbio/bc5cdr/resolve/main/CDR_Data.zip"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Anthropic's published per-token LIST RATES for the three models the Critic eval's paid arms
# (`--arm abstract/findings/direction` in `biolit_evals.critic_eval`) may run against. Taken by
# hand on 2026-08-21 from Anthropic's published pricing page.
#
# `biolit/extract/llm.py` carries a "NO PRICES HERE, DELIBERATELY" note explaining why this
# package normally keeps per-token prices OUT of the repo: a rate changes upstream and a copy
# committed here rots silently into a wrong estimate. This table is the one deliberate
# exception to that rule, not an oversight of it -- it exists so the eval's spend guard
# (`biolit_evals.critic_cost.KillSwitch`) can actually be ARMED from the CLI, rather than the
# CLI having no mechanism to price a run at all. We are consciously trading "no hardcoded rate"
# for "the guard is armed" here, because an armed guard against a rate that must be re-checked
# beats no guard at all.
#
# THIS TABLE MUST BE RE-VERIFIED AGAINST ANTHROPIC'S CURRENT PUBLISHED PRICING BEFORE EVERY
# AUTHORIZATION. Per-token rates change without notice, and a stale entry here would silently
# produce a WRONG budget bound -- not a loud one -- which is exactly the failure mode the "NO
# PRICES HERE" note exists to prevent. Do not treat this table as evergreen; do not extend it to
# a new model without independently verifying that model's rate first.
CRITIC_PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5": {"input_tokens": 5.00, "output_tokens": 25.00},
    "claude-sonnet-5": {"input_tokens": 3.00, "output_tokens": 15.00},
    "claude-haiku-4-5": {"input_tokens": 1.00, "output_tokens": 5.00},
}

# `cost_of` (`biolit_evals.critic_cost`) prices every field in `USAGE_FIELDS`, including the two
# prompt-caching fields, and raises `ValueError` if any is missing from the price map it is
# handed -- a missing price silently treated as zero is the exact undercount that module exists
# to prevent. The table above has no verified rate for either cache field: I have not confirmed
# one and will not guess one. The critic arms use no prompt caching today (verified: no
# `cache_control` anywhere under `biolit/critic/`), so in practice these two counts are zero on
# every run this prices. Both cache fields are priced at the OUTPUT rate -- the higher of the
# two known rates -- so that IF prompt caching is ever turned on without first re-verifying real
# cache rates, this OVERESTIMATES cost and the kill switch fires EARLY rather than late. Erring
# toward firing early is the safe direction for a spend guard. These are UNVERIFIED PLACEHOLDERS
# and must be replaced with real, hand-verified cache rates before enabling prompt caching on
# any critic arm.

_TOKENS_PER_MTOK = 1_000_000  # cost_of prices per TOKEN; the table above is dollars per MILLION.


def critic_prices_per_token(model: str) -> dict[str, float] | None:
    """The full 4-field per-token price map `cost_of` requires for `model`, or `None` if `model`
    has no hand-verified entry in `CRITIC_PRICES_PER_MTOK` above.

    `None` is a deliberate refusal signal, not an error to work around: the caller (the Critic
    eval CLI) is expected to refuse to run a paid arm when this returns `None`, rather than
    substitute a default or guessed price. A wrong-but-present price is worse than an explicit
    refusal, because it bounds spend against a number nobody verified.
    """
    per_mtok = CRITIC_PRICES_PER_MTOK.get(model)
    if per_mtok is None:
        return None
    output_price = per_mtok["output_tokens"] / _TOKENS_PER_MTOK
    return {
        "input_tokens": per_mtok["input_tokens"] / _TOKENS_PER_MTOK,
        "output_tokens": output_price,
        "cache_creation_input_tokens": output_price,
        "cache_read_input_tokens": output_price,
    }
