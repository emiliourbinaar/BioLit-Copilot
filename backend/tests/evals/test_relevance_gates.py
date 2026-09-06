import pytest

from biolit_evals.relevance_gates import (
    LabelledRow,
    gate1_tractability,
    gate2_controls,
    labels_hash,
    load_labels,
)

_MANIFEST = {
    "rows": {
        "r000": {"query": "q1", "cluster_key": "MESH:C1|MESH:D1", "is_distractor": False},
        "r001": {"query": "q2", "cluster_key": "MESH:C2|MESH:D2", "is_distractor": True},
    }
}
_SHEET = (
    "## 1. `r000`\n\n```\nlabel: answers\nreason: on point\n```\n\n"
    "## 2. `r001`\n\n```\nlabel: off_topic\nreason: unrelated\n```\n"
)


def test_load_labels_joins_the_sheet_to_the_manifest():
    rows = load_labels(_SHEET, _MANIFEST)

    assert [r.row_id for r in rows] == ["r000", "r001"]
    assert rows[0].cluster_key == "MESH:C1|MESH:D1"
    assert rows[0].label == "answers"
    assert rows[1].is_distractor is True


def test_load_labels_refuses_a_sheet_row_absent_from_the_manifest():
    """The manifest is the record of WHICH rows were frozen. A label for a row it does not
    know means the sheet was edited or regenerated after labelling began, and every gate
    count downstream would be computed over a row set nobody pinned."""
    sheet = _SHEET + "## 3. `r999`\n\n```\nlabel: answers\nreason: x\n```\n"

    with pytest.raises(ValueError, match="r999"):
        load_labels(sheet, _MANIFEST)


def test_load_labels_refuses_a_manifest_row_missing_from_the_sheet():
    """The mirror failure, and the more dangerous one: a silently skipped row shrinks a gate
    denominator without changing any visible verdict."""
    manifest = {
        "rows": dict(_MANIFEST["rows"], r002={"query": "q3", "cluster_key": "a|b",
                                              "is_distractor": False})
    }

    with pytest.raises(ValueError, match="r002"):
        load_labels(_SHEET, manifest)


def _rows(labels: list[str], *, distractor: bool = False) -> list[LabelledRow]:
    return [
        LabelledRow(f"r{i:03d}", "q", f"MESH:C{i}|MESH:D{i}", distractor, label, "because")
        for i, label in enumerate(labels)
    ]


def test_gate1_is_tractable_when_cant_tell_is_rare():
    result = gate1_tractability(_rows(["answers"] * 19 + ["cant_tell"]))

    assert result.verdict == "TRACTABLE"
    assert result.n_cant_tell == 1
    assert result.n == 20


def test_gate1_says_revise_schema_at_or_above_the_pre_registered_fifteen_percent():
    """The pre-registered cut, read as it falls. At >= 15% the label definitions do not fit
    the data and the whole pass stops -- Gate 3's numbers are not computed at all, because a
    filter reading over rows the annotator could not judge is not interpretable."""
    assert gate1_tractability(_rows(["cant_tell"] * 3 + ["answers"] * 17)).verdict == "REVISE_SCHEMA"
    assert gate1_tractability(_rows(["cant_tell"] * 2 + ["answers"] * 18)).verdict == "TRACTABLE"


def test_gate1_counts_only_real_rows_not_the_controls():
    """§5. The controls are a separate instrument with their own verdict; folding them into
    the tractability denominator would dilute it with rows designed to be easy."""
    rows = _rows(["cant_tell"] * 3) + _rows(["off_topic"] * 17, distractor=True)

    assert gate1_tractability(rows).n == 3


def test_gate2_is_discriminating_when_the_controls_read_off_topic():
    """ADR-0018's contribution: without this, a permissive annotator and a good filter are
    indistinguishable, and no Gate 3 reading is attributable."""
    result = gate2_controls(_rows(["off_topic"] * 7 + ["background"], distractor=True))

    assert result.verdict == "DISCRIMINATING"
    assert (result.n_off_topic, result.n) == (7, 8)


def test_gate2_is_confounded_below_the_pre_registered_seven_of_eight():
    rows = _rows(["off_topic"] * 6 + ["background"] * 2, distractor=True)

    assert gate2_controls(rows).verdict == "CONFOUNDED"


def test_labels_hash_tracks_content_not_order():
    # Seven reverse-inserted elements, per this repo's determinism-fixture convention.
    rows = _rows(["answers", "background", "off_topic", "answers", "background",
                  "off_topic", "answers"])

    assert labels_hash(rows) == labels_hash(list(reversed(rows)))
    assert labels_hash(rows) != labels_hash(rows[:-1])


def test_labels_hash_changes_when_a_label_changes():
    """It pins the JUDGMENTS, not merely which rows were shown -- `rows_hash` already does
    the latter. A gate recomputed against edited labels must not be able to claim the hash
    of the labels it was pre-registered against."""
    rows = _rows(["answers"] * 7)
    edited = _rows(["answers"] * 6 + ["off_topic"])

    assert labels_hash(rows) != labels_hash(edited)
