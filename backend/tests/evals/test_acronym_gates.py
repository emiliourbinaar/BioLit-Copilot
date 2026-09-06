import pytest

from biolit_evals.acronym_gates import (
    LabelledPair,
    gate1_tractability,
    gate2_controls,
    gate3_rate,
    gate4_type_violation,
    labels_hash,
    load_labels,
    split_by_disclosure,
)


def _rows(labels: list[str], *, control: bool = False) -> list[LabelledPair]:
    return [
        LabelledPair(f"a{i:03d}", f"S{i}", f"MESH:D{i:05d}", 1, False, control, label, "because")
        for i, label in enumerate(labels)
    ]


def test_gate1_reads_the_pre_registered_quarter_as_it_falls():
    """§5. The limit is 0.25 and NOT the relevance pass's 0.15, because ADR-0018 recorded the
    exact mistake of carrying a constant calibrated for one question into a second. 0.15 was
    calibrated for "can you judge topical relevance"; this asks "can you tell what this acronym
    means here", where nine of the 44 pairs carry no in-document gloss at all.

    Firing means REVISE_CONTEXT -- show more text per row and re-label -- not stop.
    """
    # 5/21 = 0.238 and 5/20 = 0.250 -- the pair straddles the band, so this also pins that it
    # is read AT or above rather than strictly above.
    assert gate1_tractability(_rows(["cant_tell"] * 5 + ["correct"] * 16)).verdict == "TRACTABLE"
    assert (
        gate1_tractability(_rows(["cant_tell"] * 5 + ["correct"] * 15)).verdict == "REVISE_CONTEXT"
    )


def test_gate2_is_discriminating_only_when_the_controls_are_rejected():
    """§4/§5. A control is a real surface shown against a real concept belonging to a different
    pair, so `wrong` is the only correct answer. Without this, a permissive annotator and a
    broken linker produce identical Gate 3 numbers -- ADR-0018's contribution, and the reason
    its readings were attributable where Phase 5's were not.

    ⚠️ `granularity` does NOT count as a rejection. A mispaired control is not the right
    concept at the wrong level; it is a different concept entirely, and accepting the softer
    label here would let a systematically hedging annotator clear the gate.
    """
    good = gate2_controls(_rows(["wrong"] * 7 + ["correct"], control=True))
    hedged = gate2_controls(_rows(["wrong"] * 6 + ["granularity"] * 2, control=True))

    assert (good.verdict, good.n_wrong, good.n) == ("DISCRIMINATING", 7, 8)
    assert hedged.verdict == "CONFOUNDED"


def test_gate3_reports_per_pair_and_mention_weighted_because_they_answer_different_questions():
    """§5. Per pair asks how often the mechanism produces a wrong concept; mention-weighted
    asks how much wrong text a reader actually sees. They diverge sharply here -- `APT` alone
    is 26 of the 204 mentions -- so reporting either alone would let one question's answer be
    quoted for the other.

    `cant_tell` is excluded from both denominators and reported separately: it is not a grade.
    """
    rows = [
        LabelledPair("a0", "APT", "MESH:C1", 26, True, False, "wrong", "b"),
        LabelledPair("a1", "HCC", "MESH:C2", 7, False, False, "correct", "b"),
        LabelledPair("a2", "ICH", "MESH:C3", 17, False, False, "granularity", "b"),
        LabelledPair("a3", "NAD", "MESH:C4", 1, False, False, "cant_tell", "b"),
        LabelledPair("a4", "XX", "MESH:C5", 99, False, True, "wrong", "control"),
    ]

    result = gate3_rate(rows)

    assert result.by_pair == {"wrong": 1, "correct": 1, "granularity": 1}
    assert result.by_mention == {"wrong": 26, "correct": 7, "granularity": 17}
    assert (result.n_pairs, result.n_mentions) == (3, 50)
    assert result.n_cant_tell == 1, "reported beside the rate, never inside it"


def test_gate4_is_a_contingency_table_and_computes_no_test_statistic():
    """§5/§7.1. The flag was shown to the annotator BEFORE labelling -- all ten flagged pairs
    were named in the session that produced the design -- so these labels cannot be the blind
    test of it the gate was designed to be.

    So it returns counts and nothing else. A rate, a chi-square or a lift figure here would
    read as evidence for the screening signal, when the only honest reading is "this is what
    the labels happened to say about rows the annotator already knew were flagged". DEF-0004
    rests on the mechanical inconsistency, which needs no labels at all.
    """
    rows = [
        LabelledPair("a0", "APT", "MESH:C1", 26, True, False, "wrong", "b"),
        LabelledPair("a1", "GSH", "MESH:C2", 9, True, False, "wrong", "b"),
        LabelledPair("a2", "HCC", "MESH:C3", 7, False, False, "correct", "b"),
        LabelledPair("a3", "ICH", "MESH:C4", 17, False, False, "granularity", "b"),
        LabelledPair("a4", "XX", "MESH:C5", 1, True, True, "wrong", "control"),
    ]

    result = gate4_type_violation(rows)

    assert result.flagged == {"wrong": 2}
    assert result.unflagged == {"correct": 1, "granularity": 1}
    assert not hasattr(result, "lift") and not hasattr(result, "p_value")


def test_the_disclosure_split_names_the_surfaces_rather_than_counting_them():
    """§7.1. Sixteen of the 44 pairs were named to the annotator before labelling -- nine in
    DEFECTS.md with a direction attached, and the ten type-violating ones as a table -- so
    Gate 3 is reported split rather than as one denominator that reads clean.

    The constant is a list of SURFACES fixed in the spec, not a recomputation. A split derived
    by re-deriving "which pairs did we happen to mention" after the fact would drift with
    whatever the code could still see, which is the opposite of a pre-registration.
    """
    rows = [
        LabelledPair("a0", "GSH", "MESH:C1", 9, True, False, "wrong", "b"),
        LabelledPair("a1", "DDAB", "MESH:C2", 2, False, False, "correct", "b"),
        LabelledPair("a2", "XX", "MESH:C3", 1, False, True, "wrong", "control"),
    ]

    disclosed, undisclosed = split_by_disclosure(rows)

    assert [row.surface for row in disclosed] == ["GSH"]
    assert [row.surface for row in undisclosed] == ["DDAB"], "controls belong to neither side"


_MANIFEST = {
    "rows": {
        "a000": {
            "surface": "GSH",
            "concept_id": "MESH:C563177",
            "n_mentions": 9,
            "type_violation": True,
            "is_control": False,
        }
    }
}
_SHEET = "## 1. `a000`\n\n```\nlabel: wrong\nreason: the abstract says glutathione\n```\n"


def test_load_labels_refuses_drift_between_the_sheet_and_the_frozen_manifest():
    """Both directions. A sheet row the manifest does not know means the sheet was regenerated
    after labelling began, so every count downstream would be computed over a row set nobody
    froze. A manifest row missing from the sheet is the more dangerous mirror: a skipped row
    shrinks a gate denominator without changing any visible verdict.
    """
    joined = load_labels(_SHEET, _MANIFEST)
    assert joined[0].surface == "GSH" and joined[0].type_violation is True
    assert joined[0].n_mentions == 9, "the weighting comes from the manifest, not the sheet"

    with pytest.raises(ValueError, match="a999"):
        load_labels(_SHEET + "## 2. `a999`\n\n```\nlabel: wrong\nreason: x\n```\n", _MANIFEST)

    bigger = {"rows": dict(_MANIFEST["rows"], a001={**_MANIFEST["rows"]["a000"], "surface": "CP"})}
    with pytest.raises(ValueError, match="a001"):
        load_labels(_SHEET, bigger)


def test_labels_hash_pins_the_judgments_not_merely_which_rows_were_shown():
    """`rows_hash` already pins WHICH rows were shown; this pins what was said about them, so a
    gate recomputed against edited labels cannot claim the hash it was registered against.

    Reasons are excluded deliberately: they are for a human reader, and a typo fix in one must
    not invalidate a frozen reading.
    """
    # Seven reverse-inserted elements, per this repo's determinism-fixture convention.
    rows = _rows(["correct", "wrong", "granularity", "correct", "wrong", "granularity", "correct"])
    edited = _rows(["correct", "wrong", "granularity", "correct", "wrong", "granularity", "wrong"])

    assert labels_hash(rows) == labels_hash(list(reversed(rows)))
    assert labels_hash(rows) != labels_hash(rows[:-1])
    assert labels_hash(rows) != labels_hash(edited)
