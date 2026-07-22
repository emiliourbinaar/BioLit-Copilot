import pytest

from biolit.domain.enums import EntityLabel
from biolit.ner.labels import canonical_label

CHEMICAL = EntityLabel.CHEMICAL
DISEASE = EntityLabel.DISEASE


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Chemical", CHEMICAL),
        ("chemical", CHEMICAL),
        ("B-Chemical", CHEMICAL),
        ("I-Chemical", CHEMICAL),
        ("Disease", DISEASE),
        ("B-Disease", DISEASE),
        ("I-Disease", DISEASE),
        ("O", None),
        ("Gene", None),
        ("", None),
    ],
)
def test_canonical_label(raw, expected):
    assert canonical_label(raw) == expected


def test_canonical_label_returns_enum_member():
    # Not just string-equal: the return type is the EntityLabel enum, so downstream
    # code that pattern-matches on the type (not the raw string) stays sound.
    assert isinstance(canonical_label("Chemical"), EntityLabel)
