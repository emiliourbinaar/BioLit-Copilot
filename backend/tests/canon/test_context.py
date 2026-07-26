from biolit.canon.context import check_document_context
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

CHEMICAL = EntityLabel.CHEMICAL

_ABSTRACT = (
    "Torsades de pointes (TdP) is a known complication of QT-prolonging drugs. "
    "We reviewed 240 consecutive admissions over a four-year period and identified "
    "eighteen patients in whom TdP followed the administration of metformin or a "
    "related agent. All episodes resolved after withdrawal of the offending drug."
)


def test_a_single_sentence_is_rejected_as_insufficient_context():
    # THE case this guard exists for. A sentence and its entities are perfectly
    # self-consistent, so no bounds or alignment check can catch it -- only the shape of
    # the text itself. This is exactly what the domain eval passes as a "document".
    sentence = "Patients receiving metformin developed TdP."
    entities = [Entity(text="metformin", label=CHEMICAL, start=19, end=28)]
    check = check_document_context(entities, sentence)
    assert check.ok is False
    assert "insufficient context" in (check.reason or "")


def test_span_that_does_not_slice_to_the_entity_text_is_a_hard_violation():
    # In bounds but misaligned: the text is a different document, or the offsets were
    # computed against a differently-preprocessed copy. Bounds alone would not catch it.
    entities = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    check = check_document_context(entities, _ABSTRACT)
    assert check.ok is False
    assert "slice" in (check.reason or "")


def test_a_real_abstract_passes():
    # The guard must not fire on correct usage, or it would suppress the mechanism it
    # exists to protect.
    i = _ABSTRACT.index("metformin")
    entities = [Entity(text="metformin", label=CHEMICAL, start=i, end=i + 9)]
    assert check_document_context(entities, _ABSTRACT).ok is True


def test_span_beyond_the_end_of_text_is_a_hard_violation():
    # Provable: these offsets cannot refer to this string, so the caller passed text that
    # is not the document the entities were extracted from.
    entities = [Entity(text="metformin", label=CHEMICAL, start=5000, end=5009)]
    check = check_document_context(entities, _ABSTRACT)
    assert check.ok is False
    assert "outside" in (check.reason or "")
