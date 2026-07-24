import json

from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit_evals.canon_eval import run_canon_eval
from biolit_evals.mesh_gold import GoldMention

CHEM = EntityLabel.CHEMICAL

_EXPECTED_KEYS = {
    "timestamp",
    "git_sha",
    "dataset",
    "artifact_source",
    "n_aliases",
    "n",
    "correct",
    "linked",
    "precision",
    "recall",
    "f1",
    "nil_rate",
    "tiebreak_rate",
}


def test_run_canon_eval_scores_and_appends_one_log_line(tmp_path):
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    linker = DictionaryLinker(MeshDictionary({"metformin": [AliasEntry(met, True)]}))
    gold = [
        GoldMention(
            pmid="1", start=0, end=9, text="Metformin", label=CHEM, mesh_ids=("MESH:D008687",)
        )
    ]
    log = tmp_path / "nested" / "canon_runs.jsonl"  # parent must be created
    metrics = run_canon_eval(
        gold=gold,
        linker=linker,
        dataset="fixture",
        artifact_source="fixture",
        n_aliases=123,
        log_path=str(log),
        git_sha="abc1234",
        now="2026-07-22T00:00:00+00:00",
    )
    assert metrics.correct == 1
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert set(json.loads(lines[0]).keys()) == _EXPECTED_KEYS
