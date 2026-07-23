import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from biolit.canon.linker import DictionaryLinker, Linker
from biolit_evals.linking_scoring import LinkingMetrics, score_linking
from biolit_evals.mesh_gold import GoldMention

DEFAULT_LOG = "evals/canon_runs.jsonl"
DOMAIN_NORM_GOLD = "evals/gold/domain_normalization_sample.jsonl"


def run_canon_eval(
    *,
    gold: list[GoldMention],
    linker: Linker,
    dataset: str,
    artifact_source: str,
    log_path: str,
    git_sha: str,
    now: str,
) -> LinkingMetrics:
    """Score linking on `gold` and append one JSON line to `log_path`. All impure inputs
    (gold, linker, log_path, git_sha, now) are injected so this stays offline-testable."""
    metrics = score_linking(gold, linker)
    line = {
        "timestamp": now,
        "git_sha": git_sha,
        "dataset": dataset,
        "artifact_source": artifact_source,
        "n": metrics.n,
        "correct": metrics.correct,
        "linked": metrics.linked,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "nil_rate": metrics.nil_rate,
        "tiebreak_rate": metrics.tiebreak_rate,
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return metrics


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # best-effort metadata; never fail an eval over it
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    # Heavy imports (artifact load, gold download) are local so importing this module for
    # `run_canon_eval` stays cheap and offline.
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit_evals.mesh_gold import load_domain_norm_sample
    from biolit_evals.mesh_gold_download import load_bc5cdr_norm_gold

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    linker = DictionaryLinker(MeshDictionary.from_artifact(settings.mesh_artifact_path))
    if args.dataset == "bc5cdr":
        gold = load_bc5cdr_norm_gold(settings.bc5cdr_cdr_zip_url)
    else:
        gold = load_domain_norm_sample(DOMAIN_NORM_GOLD)

    metrics = run_canon_eval(
        gold=gold,
        linker=linker,
        dataset=args.dataset,
        artifact_source=settings.mesh_artifact_path,
        log_path=DEFAULT_LOG,
        git_sha=_git_sha(),
        now=datetime.now(UTC).isoformat(),
    )
    print(
        f"{args.dataset}: P={metrics.precision:.4f} R={metrics.recall:.4f} F1={metrics.f1:.4f} "
        f"(correct={metrics.correct}/{metrics.n}, linked={metrics.linked}, "
        f"NIL={metrics.nil_rate:.3f}, tiebreak={metrics.tiebreak_rate:.3f})"
    )


if __name__ == "__main__":
    main()
