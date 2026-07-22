import json

from biolit.domain.records import Entity
from biolit.ner.model import NerModel
from biolit_evals.ner_eval import run_eval


def test_run_eval_scores_and_appends_log(tmp_path):
    # Gold examples: (text, gold_entities). The fake model predicts from a lookup by text.
    examples = [
        ("metformin in PCOS", [Entity(text="metformin", label="CHEMICAL", start=0, end=9)]),
    ]
    preds = {
        "metformin in PCOS": [
            {"entity_group": "Chemical", "score": 0.99, "word": "metformin", "start": 0, "end": 9}
        ]
    }
    model = NerModel(predictor=lambda text: preds.get(text, []))
    log = tmp_path / "runs.jsonl"

    prf = run_eval(
        examples,
        model,
        dataset="synthetic",
        split="test",
        model_id="fake",
        log_path=str(log),
        git_sha="abc1234",
        now="2026-07-21T00:00:00Z",
    )
    assert prf.f1 == 1.0
    line = json.loads(log.read_text(encoding="utf-8").strip())
    assert line["dataset"] == "synthetic"
    assert line["f1"] == 1.0
    assert line["n_examples"] == 1
    assert line["git_sha"] == "abc1234"
    assert line["model_id"] == "fake"
    assert {"timestamp", "precision", "recall", "tp", "fp", "fn"} <= line.keys()


def test_run_eval_appends_not_overwrites(tmp_path):
    log = tmp_path / "runs.jsonl"
    model = NerModel(predictor=lambda text: [])
    run_eval(
        [("x", [])],
        model,
        dataset="a",
        split="test",
        model_id="m",
        log_path=str(log),
        git_sha="s",
        now="t",
    )
    run_eval(
        [("x", [])],
        model,
        dataset="b",
        split="test",
        model_id="m",
        log_path=str(log),
        git_sha="s",
        now="t",
    )
    assert len([ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]) == 2
