import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from biolit.config import get_settings
from biolit.domain.records import Entity
from biolit.ner.extract import extract_entities
from biolit.ner.model import NerModel
from biolit_evals.datasets import load_bc5cdr_test, load_domain_sample
from biolit_evals.scoring import PRF, score_corpus

# Paths below are relative to the `backend/` directory, which is where this
# project's commands are run from.
DEFAULT_LOG = "evals/runs.jsonl"
DOMAIN_GOLD = "evals/gold/domain_sample.jsonl"


def run_eval(
    examples: list[tuple[str, list[Entity]]],
    model: NerModel,
    *,
    dataset: str,
    split: str,
    model_id: str,
    log_path: str,
    git_sha: str,
    now: str,
    score_threshold: float = 0.5,
) -> PRF:
    """Run NER extraction over `examples`, score it, and append one JSON line to the log.

    `examples`, `model`, `log_path`, `git_sha`, and `now` are all injected by the caller
    so this function stays testable offline (no filesystem discovery, no live git call,
    no wall-clock read). Creates the log file's parent directory if it does not already
    exist, then appends (never truncates or rewrites) a single JSON record summarizing
    the run's precision/recall/F1 and counts.
    """
    pairs: list[tuple[list[Entity], list[Entity]]] = []
    for text, gold in examples:
        pred = extract_entities(text, model, score_threshold=score_threshold)
        pairs.append((gold, pred))
    prf = score_corpus(pairs)
    line = {
        "timestamp": now,
        "model_id": model_id,
        "dataset": dataset,
        "split": split,
        "precision": prf.precision,
        "recall": prf.recall,
        "f1": prf.f1,
        "tp": prf.tp,
        "fp": prf.fp,
        "fn": prf.fn,
        "n_examples": len(examples),
        "git_sha": git_sha,
    }
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line) + "\n")
    return prf


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # sha is best-effort metadata; never fail an eval over it
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bc5cdr", "domain"], required=True)
    args = parser.parse_args(argv)

    settings = get_settings()
    model = NerModel.load(settings)
    if args.dataset == "bc5cdr":
        examples = load_bc5cdr_test()
        split = "test"
    else:
        examples = load_domain_sample(DOMAIN_GOLD)
        split = "domain"

    prf = run_eval(
        examples,
        model,
        dataset=args.dataset,
        split=split,
        model_id=settings.ner_model_id,
        log_path=DEFAULT_LOG,
        git_sha=_git_sha(),
        now=datetime.now(UTC).isoformat(),
        score_threshold=settings.ner_score_threshold,
    )
    print(
        f"{args.dataset}: P={prf.precision:.4f} R={prf.recall:.4f} F1={prf.f1:.4f} "
        f"(tp={prf.tp} fp={prf.fp} fn={prf.fn}, n={len(examples)})"
    )


if __name__ == "__main__":
    main()
