import pytest

from biolit.domain.records import Cluster
from biolit_evals.cluster_eval import (
    Workload,
    assert_gold_cluster_anchor,
    assert_key_recall_anchor,
    cluster_key_metrics,
    gold_clusters_from_relations,
    key_metrics,
    paper_pair_metrics,
    paper_pairs,
    workload,
)
from biolit_evals.end_to_end import metrics_from_counts


def test_key_metrics_score_predicted_pairs_per_document_against_gold_cid():
    gold = {"1": {("MESH:D008687", "MESH:D011085")}}
    pred = {"1": {("MESH:D008687", "MESH:D011085"), ("MESH:D001241", "MESH:D011085")}}
    m = key_metrics(pred, gold)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.recall == 1.0


def test_paper_pair_metrics_are_not_an_alias_of_cluster_key_metrics():
    # Same key, but the prediction recovers only 2 of the 3 papers in the gold cluster.
    # Cluster-key level cannot see the miss (the key matches exactly, F1 == 1.0);
    # paper-pair level does (2 of 3 gold pairs missing, recall 1/3).
    # A paper_pair_metrics that secretly computes cluster-key agreement returns F1 1.0
    # here and fails.
    gold = [Cluster(key="MESH:D008687|MESH:D011085", paper_ids=["A", "B", "C"])]
    pred = [Cluster(key="MESH:D008687|MESH:D011085", paper_ids=["A", "B"])]

    ck = cluster_key_metrics(pred, gold)
    assert ck.f1 == 1.0  # blind to the missing paper

    pp = paper_pair_metrics(pred, gold)
    assert (pp.tp, pp.fp, pp.fn) == (1, 0, 2)
    assert pp.precision == 1.0
    assert pp.recall == 1 / 3
    assert pp.f1 == 0.5


def test_gold_clusters_from_relations_aggregates_shared_pairs_and_drops_singletons():
    # pmids "1" and "2" share one relation pair -> one cluster holding both pmids.
    # pmid "3" holds a pair no one else has -> exactly one document, dropped at the
    # >= min_size boundary (a `> min_size` mutant would also drop the shared pair, since
    # its count is exactly 2).
    # The key is chemical|disease built from the tuple in order -- a swapped _key would
    # produce "MESH:D011085|MESH:D008687" instead and fail the exact-equality check below.
    relations = {
        "1": {("MESH:D008687", "MESH:D011085")},
        "2": {("MESH:D008687", "MESH:D011085")},
        "3": {("MESH:D001241", "MESH:D011085")},
    }
    clusters = gold_clusters_from_relations(relations)
    assert clusters == [Cluster(key="MESH:D008687|MESH:D011085", paper_ids=["1", "2"])]


def test_key_metrics_covers_pmids_present_on_only_one_side():
    # pmid "1" is shared -> tp. pmid "2" is gold-only (pred never produced anything for it,
    # e.g. NER missed the whole document) -> its pair must land in fn. pmid "3" is pred-only
    # (pred hallucinated a document gold has no relation for) -> its pair must land in fp.
    # An `&` mutant on `set(pred) | set(gold)` would iterate only pmid "1", silently dropping
    # both the gold-only fn and the pred-only fp.
    gold = {
        "1": {("MESH:D008687", "MESH:D011085")},
        "2": {("MESH:D001241", "MESH:D009325")},
    }
    pred = {
        "1": {("MESH:D008687", "MESH:D011085")},
        "3": {("MESH:D007328", "MESH:D014456")},
    }
    m = key_metrics(pred, gold)
    assert (m.tp, m.fp, m.fn) == (1, 1, 1)


def test_paper_pairs_are_unordered_sorted_2tuples():
    clusters = [Cluster(key="k", paper_ids=["B", "A", "C"])]
    assert paper_pairs(clusters) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_top5_share_exposes_one_oversized_cluster_dominating_the_critic_budget():
    # One 25-paper cluster is 300 comparisons on its own; five 2-paper clusters are 5.
    # An aggregate pair count cannot show that concentration; this can.
    clusters = [Cluster(key=f"k{i}", paper_ids=[f"p{i}_{j}" for j in range(2)]) for i in range(5)]
    clusters.append(Cluster(key="big", paper_ids=[f"b{j}" for j in range(25)]))
    w = workload(clusters)
    assert w.n_clusters == 6
    assert w.n_paper_pairs == 305
    assert w.largest_cluster == 25
    assert round(w.top5_pair_share, 4) == round(304 / 305, 4)


def test_workload_handles_empty_cluster_list():
    # Both guards exist specifically to handle empty case. Dropping either would raise
    # IndexError (sizes[0]) or ZeroDivisionError (total) undetected. cluster_papers
    # can return [] when no chemical|disease key is shared by two papers.
    w = workload([])
    assert w == Workload(n_clusters=0, n_paper_pairs=0, largest_cluster=0, top5_pair_share=0.0)


def test_the_key_recall_anchor_raises_when_cross_product_misses_a_gold_pair():
    # Cross-product on gold entities cannot miss a gold pair -- both endpoints are
    # annotated. Recall below 1.0 means the harness is wrong (gold parsing, MeSH id
    # prefixing, label assignment), not that the number is interesting.
    assert_key_recall_anchor(metrics_from_counts(10, 5, 0), arm="A")  # recall 1.0, fine
    with pytest.raises(SystemExit, match="key recall"):
        assert_key_recall_anchor(metrics_from_counts(9, 5, 1), arm="A")


def test_the_key_recall_anchor_catches_even_small_misses():
    # A small miss (recall 0.9990 from tp=999, fn=1) still violates the anchor.
    # This discriminates against a loosened tolerance like `round(recall, 2)` or
    # `abs(recall - 1.0) < 0.05`, which would incorrectly pass a near-miss harness.
    with pytest.raises(SystemExit, match="key recall"):
        assert_key_recall_anchor(metrics_from_counts(999, 0, 1), arm="A")


def test_the_gold_cluster_anchor_checks_the_LOADER_not_clustering_quality():
    assert_gold_cluster_anchor(500, 80)
    assert_gold_cluster_anchor(1500, 325)
    with pytest.raises(SystemExit, match="gold cluster"):
        assert_gold_cluster_anchor(500, 79)


def test_the_gold_cluster_anchor_skips_untabulated_corpus_sizes():
    # The anchor table only knows about 500 and 1500 documents. For any other size,
    # the anchor returns silently (does not raise). This discriminates against a mutant
    # that deletes the `if expected is None: return` guard, which would always raise.
    assert_gold_cluster_anchor(3, 99)  # Untabulated size, any count is OK
    assert_gold_cluster_anchor(100, 0)  # Another untabulated size, zero is OK
