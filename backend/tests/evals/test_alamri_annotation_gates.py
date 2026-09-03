import pytest

from biolit.domain.records import ContradictionLabel
from biolit_evals.alamri_annotation_gates import LabelRow, evaluate_strictness, join_labels
from biolit_evals.alamri_gold import AlamriPair
from biolit_evals.annotation_export import Annotation


def _pair(pair_id: str, stratum: str, label: ContradictionLabel) -> AlamriPair:
    return AlamriPair(
        pair_id=pair_id,
        question_id_a="qa",
        question_id_b="qb",
        paper_id_a="1",
        paper_id_b="2",
        label=label,
        stratum=stratum,
        signals=(),
    )


def test_join_labels_refuses_a_batch_the_annotations_do_not_cover():
    """A silently dropped row moves a gate verdict directly: a missing `contradiction`
    lowers g, pushing Gate 2 toward STOP -- the verdict that retires the design."""
    batch = [
        _pair("p1", "clean", ContradictionLabel.contradiction),
        _pair("p2", "clean", ContradictionLabel.contradiction),
    ]
    annotations = {"p1": Annotation(label="contradiction", reason="r")}
    with pytest.raises(ValueError, match="p2"):
        join_labels(annotations, batch)


def test_join_labels_refuses_an_annotation_for_a_pair_not_in_the_batch():
    batch = [_pair("p1", "clean", ContradictionLabel.contradiction)]
    annotations = {
        "p1": Annotation(label="contradiction", reason="r"),
        "p9": Annotation(label="agreement", reason="r"),
    }
    with pytest.raises(ValueError, match="p9"):
        join_labels(annotations, batch)


def _rows(stratum, labels):
    return [LabelRow(f"{stratum}{i}", "x", label, stratum, "r") for i, label in enumerate(labels)]


A_CLEAN = ["agreement"] * 8 + ["insufficient_overlap"] * 2
D_CLEAN = ["insufficient_overlap"] * 5


def test_distractors_marked_unrelated_and_agreements_marked_related_reads_discriminating():
    """The reading that lets pi-hat stand on its own: the annotator uses
    insufficient_overlap for its meaning rather than as a general escape hatch."""
    read = evaluate_strictness(_rows("agreement", A_CLEAN), _rows("distractor", D_CLEAN))
    assert read.verdict == "DISCRIMINATING"


def test_both_controls_marked_unrelated_reads_strictness_confounded():
    """ADR-0017's defect (b): Phase 5 applied insufficient_overlap to 4 of 8 gold agreement
    pairs, so the pull was general. At >= 5/10 here the pi-hat readings are confounded in
    the same direction and may not be reported as clean validity estimates."""
    read = evaluate_strictness(
        _rows("agreement", ["insufficient_overlap"] * 5 + ["agreement"] * 5),
        _rows("distractor", D_CLEAN),
    )
    assert read.verdict == "STRICTNESS_CONFOUNDED"


def test_distractors_not_marked_unrelated_reads_uninformative():
    """If the known-unrelated anchor is not called unrelated, the label is not being used
    for its meaning and the whole strictness read says nothing -- whatever the agreement
    controls happen to show."""
    read = evaluate_strictness(
        _rows("agreement", A_CLEAN),
        _rows("distractor", ["insufficient_overlap"] * 3 + ["agreement"] * 2),
    )
    assert read.verdict == "UNINFORMATIVE"


def test_agreement_controls_read_as_contradictions_are_flagged_not_folded_in():
    """Not strictness -- it would mean the derived `agreement` label is itself wrong. The
    instruction is to flag an unanticipated pattern rather than round it to the nearest
    named reading."""
    read = evaluate_strictness(
        _rows("agreement", ["contradiction"] * 7 + ["agreement"] * 3),
        _rows("distractor", D_CLEAN),
    )
    assert read.verdict == "UNEXPECTED"
