import pytest

from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit_evals.linking_scoring import score_linking
from biolit_evals.mesh_gold import GoldMention

CHEM = EntityLabel.CHEMICAL


def _gold(text, ids):
    return GoldMention(pmid="1", start=0, end=len(text), text=text, label=CHEM, mesh_ids=ids)


def _linker():
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    asp = MeshConcept(id="MESH:D001241", name="Aspirin")
    ambig_a = MeshConcept(id="MESH:D000001", name="A")
    ambig_b = MeshConcept(id="MESH:D000002", name="B")
    return DictionaryLinker(
        MeshDictionary(
            {
                "metformin": [AliasEntry(met, True)],
                "aspirin": [AliasEntry(asp, True)],
                "ambig": [AliasEntry(ambig_a, False), AliasEntry(ambig_b, False)],
            }
        )
    )


def test_score_linking_computes_pr_f1_nil_and_tiebreak():
    gold = [
        _gold("Metformin", ("MESH:D008687",)),  # correct
        _gold("Aspirin", ("MESH:D999999",)),  # linked but wrong id
        _gold("Unknownium", ("MESH:D111111",)),  # NIL
        _gold("ambig", ("MESH:D000001",)),  # correct via tiebreak (smallest id)
        _gold("Unlinkable", ()),  # excluded from the eval set entirely
    ]
    m = score_linking(gold, _linker())
    assert m.n == 4  # the unlinkable gold mention is dropped
    assert m.correct == 2  # Metformin + ambig
    assert m.linked == 3  # Metformin, Aspirin, ambig (Unknownium is NIL)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 4)
    assert m.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert m.nil_rate == pytest.approx(1 / 4)
    assert m.tiebreak_rate == pytest.approx(1 / 4)


def test_score_linking_empty_set_is_all_zero():
    m = score_linking([], _linker())
    assert (m.n, m.correct, m.linked) == (0, 0, 0)
    assert (m.precision, m.recall, m.f1, m.nil_rate, m.tiebreak_rate) == (0.0, 0.0, 0.0, 0.0, 0.0)
