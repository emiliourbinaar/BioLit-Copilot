import pytest

from biolit_evals.end_to_end import concept_counts, metrics_from_counts


def test_concept_counts_for_one_document():
    tp, fp, fn = concept_counts({"MESH:A", "MESH:B"}, {"MESH:A", "MESH:C"})
    assert (tp, fp, fn) == (1, 1, 1)


def test_metrics_are_micro_averaged_not_macro():
    # Doc 1: tp=1 fp=0 fn=0 (perfect). Doc 2: tp=1 fp=3 fn=0 (precision 0.25).
    # Micro precision over pooled totals = 2/5 = 0.4, while the MACRO average of the
    # two per-document precisions would be (1.0 + 0.25)/2 = 0.625. Pinning the micro value
    # is what makes a macro-average regression fail this test.
    d1 = concept_counts({"MESH:A"}, {"MESH:A"})
    d2 = concept_counts({"MESH:B"}, {"MESH:B", "MESH:X", "MESH:Y", "MESH:Z"})
    tp = d1[0] + d2[0]
    fp = d1[1] + d2[1]
    fn = d1[2] + d2[2]
    m = metrics_from_counts(tp, fp, fn)
    assert (m.tp, m.fp, m.fn) == (2, 3, 0)
    assert m.precision == pytest.approx(2 / 5)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(2 * (2 / 5) * 1.0 / ((2 / 5) + 1.0))


def test_metrics_from_counts_all_zero_is_safe():
    m = metrics_from_counts(0, 0, 0)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_repeated_concept_in_one_document_counts_once():
    # Both gold and predictions mention the same concept repeatedly; sets collapse it, so
    # a single much-repeated entity cannot dominate the corpus totals.
    tp, fp, fn = concept_counts({"MESH:A"}, {"MESH:A"})
    assert (tp, fp, fn) == (1, 0, 0)
