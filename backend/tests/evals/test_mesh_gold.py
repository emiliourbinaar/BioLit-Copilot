import json
from pathlib import Path

import pytest

from biolit.domain.enums import EntityLabel
from biolit_evals.mesh_gold import load_domain_norm_sample, parse_pubtator, reconcile_mesh_id

_FIX = Path(__file__).parent / "fixtures"


def test_reconcile_prefixes_bare_and_preserves_prefixed_and_drops_unlinkable():
    assert reconcile_mesh_id("D008687") == ("MESH:D008687",)
    assert reconcile_mesh_id("MESH:D011085") == ("MESH:D011085",)
    assert reconcile_mesh_id("OMIM:125853") == ("OMIM:125853",)
    assert reconcile_mesh_id("-1") == ()
    assert reconcile_mesh_id("D1|D2") == ("MESH:D1", "MESH:D2")


def test_parse_pubtator_extracts_mentions_and_skips_non_target_types():
    mentions = parse_pubtator((_FIX / "pubtator_sample.txt").read_text(encoding="utf-8"))
    assert len(mentions) == 3  # Gene row skipped
    met = mentions[0]
    assert met.text == "Metformin" and met.label == EntityLabel.CHEMICAL
    assert met.mesh_ids == ("MESH:D008687",)
    assert mentions[2].mesh_ids == ()  # CFD, unlinkable (-1)


def test_load_domain_norm_sample_parses_mesh_ids():
    mentions = load_domain_norm_sample(str(_FIX / "domain_norm_fixture.jsonl"))
    assert [m.mesh_ids for m in mentions] == [("MESH:D008687",), ("MESH:D011085",)]


def test_load_domain_norm_sample_rejects_span_text_mismatch(tmp_path):
    # An annotator off-by-one: offsets in bounds but pointing at the wrong substring.
    # Silently trusting them would corrupt the normalization gold with no downstream signal.
    rec = {
        "pmid": "1",
        "text": "Metformin treats PCOS",
        "entities": [
            {"start": 0, "end": 9, "label": "CHEMICAL", "text": "WRONG", "mesh_id": "D008687"}
        ],
    }
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_domain_norm_sample(str(bad))
