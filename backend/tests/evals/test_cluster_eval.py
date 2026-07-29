from biolit.domain.records import Cluster
from biolit_evals.cluster_eval import (
    Workload,
    cluster_key_metrics,
    gold_clusters_from_relations,
    key_metrics,
    paper_pair_metrics,
    paper_pairs,
    workload,
)


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
