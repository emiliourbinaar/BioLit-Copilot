from pathlib import Path

from biolit_evals.datasets import load_domain_sample

GOLD = Path(__file__).parents[2] / "evals" / "gold" / "domain_sample.jsonl"


def test_domain_gold_loads_and_is_wellformed():
    pairs = load_domain_sample(str(GOLD))
    assert len(pairs) >= 30  # target sample size
    total_entities = 0
    for text, ents in pairs:
        for e in ents:
            assert e.label in ("CHEMICAL", "DISEASE")
            assert e.start is not None and e.end is not None
            assert text[e.start : e.end] == e.text  # offsets align
            total_entities += 1
    assert total_entities > 0  # sample is not vacuous
