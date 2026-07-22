import pytest

from biolit.ner.labels import CHEMICAL, DISEASE, canonical_label


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
