import pytest

from biolit.cluster.group import pairing_diagnostics
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Cluster, Entity, ExtractedRecord
from biolit_evals.cluster_eval import (
    Workload,
    assert_gold_cluster_anchor,
    assert_key_recall_anchor,
    cluster_key_metrics,
    gold_clusters_from_relations,
    key_metrics,
    paper_pair_metrics,
    paper_pairs,
    run_cluster_eval,
    synthesize_records,
    workload,
)
from biolit_evals.end_to_end import metrics_from_counts
from biolit_evals.mesh_gold import GoldDocument, GoldMention


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


def test_gold_cluster_pmids_and_keys_are_both_sorted_deterministically():
    # gold_clusters_from_relations has the SAME two ordering guarantees as cluster_papers
    # (sorted keys, sorted pmids) and the same weak-fixture problem: the 2-pmid fixture above
    # cannot catch a dropped sort reliably. Seven pmids inserted in reverse -> 1/5040.
    # THREE keys, also inserted in reverse, pin the key ordering that no fixture observed
    # before: only one key ever cleared min_size, so a broken key sort was unobservable.
    pmids = ["7", "6", "5", "4", "3", "2", "1"]
    shared = ("MESH:D008687", "MESH:D011085")
    relations: dict[str, set[tuple[str, str]]] = {p: {shared} for p in pmids}
    for p in ("1", "2"):
        relations[p] |= {("MESH:D001241", "MESH:D014456"), ("MESH:C000657245", "MESH:D003924")}
    clusters = gold_clusters_from_relations(relations)
    assert [c.key for c in clusters] == [
        "MESH:C000657245|MESH:D003924",
        "MESH:D001241|MESH:D014456",
        "MESH:D008687|MESH:D011085",
    ]
    assert clusters[2].paper_ids == ["1", "2", "3", "4", "5", "6", "7"]


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
    assert w.n_cluster_comparisons == 305
    # Disjoint paper sets, so the two totals coincide here. Asserted deliberately: this
    # coincidence is why a single field hid the multiplicity bug, and the discriminating
    # test below is the one that separates them.
    assert w.n_distinct_paper_pairs == 305
    assert w.largest_cluster == 25
    assert round(w.top5_pair_share, 4) == round(304 / 305, 4)


def test_workload_separates_critic_calls_from_cluster_comparisons():
    # THE DISCRIMINATING TEST for the two quantities. The same paper pair placed under two
    # keys is TWO cluster comparisons but ONE Critic call: ContradictionFinding is
    # (paper_id_a, paper_id_b, label, rationale) with no key field, so the pair is the unit
    # of work and cannot be reported twice. Every other workload fixture in this file uses
    # disjoint paper sets, where both quantities coincide -- which is exactly how a single
    # field labelled "n_paper_pairs" carried the multiplicity count undetected.
    clusters = [
        Cluster(key="MESH:D000001|MESH:D000002", paper_ids=["P1", "P2"]),
        Cluster(key="MESH:D000003|MESH:D000004", paper_ids=["P1", "P2"]),
    ]
    w = workload(clusters)
    assert w.n_cluster_comparisons == 2
    assert w.n_distinct_paper_pairs == 1


def test_workload_handles_empty_cluster_list():
    # Both guards exist specifically to handle empty case. Dropping either would raise
    # IndexError (sizes[0]) or ZeroDivisionError (total) undetected. cluster_papers
    # can return [] when no chemical|disease key is shared by two papers.
    w = workload([])
    assert w == Workload(
        n_clusters=0,
        n_cluster_comparisons=0,
        n_distinct_paper_pairs=0,
        largest_cluster=0,
        top5_pair_share=0.0,
    )


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


def test_a_multi_id_gold_mention_becomes_one_entity_per_id():
    # INTERFACE DECISION: Entity.canonical_id stays str|None; each (mention, id) is its own
    # Entity at the same span. Packing ids into one string would collide with "|", the
    # cluster-key delimiter, and push id-set logic into the pairing code both arms share.
    doc = GoldDocument(
        pmid="1",
        text="Metformin and nausea.",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="Metformin",
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=14,
                end=20,
                text="nausea",
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D009325", "MESH:D012640"),
            ),
        ],
    )
    records, texts = synthesize_records([doc])
    assert len(records) == 1
    ids = [(e.label, e.canonical_id, e.start, e.end, e.text) for e in records[0].entities]
    assert ids == [
        (EntityLabel.CHEMICAL, "MESH:D008687", 0, 9, "Metformin"),
        (EntityLabel.DISEASE, "MESH:D009325", 14, 20, "nausea"),
        (EntityLabel.DISEASE, "MESH:D012640", 14, 20, "nausea"),
    ]
    assert texts == {"1": "Metformin and nausea."}


def test_a_gold_mention_with_no_mesh_id_becomes_one_NIL_entity_not_dropped():
    # A mention with mesh_ids=() (BC5CDR's unlinkable/-1 case) must still surface as an
    # Entity(canonical_id=None), not vanish. The list comprehension iterates mesh_ids per
    # mention, so an empty tuple naively yields zero Entity objects for that mention --
    # invisible to pairing_diagnostics, which can only count NIL mentions that exist.
    doc = GoldDocument(
        pmid="1",
        text="Metformin and rash.",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="Metformin",
                label=EntityLabel.CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=14,
                end=18,
                text="rash",
                label=EntityLabel.DISEASE,
                mesh_ids=(),
            ),
        ],
    )
    records, texts = synthesize_records([doc])
    assert len(records[0].entities) == 2
    nil_entity = records[0].entities[1]
    assert nil_entity.canonical_id is None
    assert nil_entity.label is EntityLabel.DISEASE
    assert nil_entity.text == "rash"
    assert nil_entity.start == 14
    assert nil_entity.end == 18

    diag = pairing_diagnostics(records, texts=texts)
    assert diag.nil_disease_mentions == 1
    assert diag.nil_chemical_mentions == 0


def test_run_cluster_eval_scores_both_strategies_and_writes_one_log_line(tmp_path):
    # Two documents sharing one gold CID pair -> exactly one gold cluster of size 2.
    docs = [
        GoldDocument(
            pmid=p,
            text="Metformin caused nausea.",
            mentions=[
                GoldMention(
                    pmid=p,
                    start=0,
                    end=9,
                    text="Metformin",
                    label=EntityLabel.CHEMICAL,
                    mesh_ids=("MESH:D008687",),
                ),
                GoldMention(
                    pmid=p,
                    start=17,
                    end=23,
                    text="nausea",
                    label=EntityLabel.DISEASE,
                    mesh_ids=("MESH:D009325",),
                ),
            ],
        )
        for p in ("1", "2")
    ]
    relations = {p: {("MESH:D008687", "MESH:D009325")} for p in ("1", "2")}
    log = tmp_path / "cluster_runs.jsonl"
    result = run_cluster_eval(
        documents=docs,
        relations=relations,
        arm="A",
        dataset="unit",
        log_path=str(log),
        git_sha="deadbee",
        now="2026-07-28T00:00:00+00:00",
    )
    assert result["n_gold_clusters"] == 1
    for strategy in ("cross_product", "same_sentence"):
        assert result["strategies"][strategy]["paper_pair"]["f1"] == 1.0
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_a_named_corpus_whose_document_count_is_wrong_halts_before_writing(tmp_path):
    # CLOSES A GAP THE GOLD-CLUSTER ANCHOR LEAVES OPEN. That anchor returns silently for
    # untabulated document counts (deliberately -- unit fixtures rely on it), so an Arm B run
    # over anything other than exactly 500 documents was scored and LOGGED with no anchor
    # firing at all: the key-recall anchor is Arm-A-only by construction, and `dataset` was a
    # hardcoded string never checked against reality. Here the tag claims 500 and 2 arrived.
    docs = [
        GoldDocument(
            pmid=p,
            text="Metformin caused nausea.",
            mentions=[
                GoldMention(
                    pmid=p,
                    start=0,
                    end=9,
                    text="Metformin",
                    label=EntityLabel.CHEMICAL,
                    mesh_ids=("MESH:D008687",),
                ),
            ],
        )
        for p in ("1", "2")
    ]
    log = tmp_path / "cluster_runs.jsonl"
    with pytest.raises(SystemExit, match="declares 500 documents"):
        run_cluster_eval(
            documents=docs,
            relations={},
            arm="B",
            dataset="bc5cdr_test500",
            log_path=str(log),
            git_sha="deadbee",
            now="2026-07-28T00:00:00+00:00",
        )
    # Nothing may be appended: a halted run must not leave a partial result behind.
    assert not log.exists()


def test_run_cluster_eval_scores_arm_b_from_records_and_texts_directly(tmp_path):
    # Arm B never passes documents=; it passes records=+texts= from the real pipeline. This
    # is the only arm that measures the production path, so its entry must be covered
    # directly rather than only exercised transitively through documents= (Arm A). It also
    # pins that the injected impure inputs (now/git_sha/dataset/arm) actually land in the
    # written line -- a run_cluster_eval that ignored them would still pass a test that
    # checks only metric shape.
    records = [
        ExtractedRecord(
            paper_id=p,
            entities=[
                Entity(
                    text="Metformin",
                    label=EntityLabel.CHEMICAL,
                    start=0,
                    end=9,
                    canonical_id="MESH:D008687",
                ),
                Entity(
                    text="nausea",
                    label=EntityLabel.DISEASE,
                    start=17,
                    end=23,
                    canonical_id="MESH:D009325",
                ),
            ],
        )
        for p in ("1", "2")
    ]
    texts = {p: "Metformin caused nausea." for p in ("1", "2")}
    relations = {p: {("MESH:D008687", "MESH:D009325")} for p in ("1", "2")}
    log = tmp_path / "cluster_runs_b.jsonl"

    result = run_cluster_eval(
        records=records,
        texts=texts,
        relations=relations,
        arm="B",
        dataset="unit_b",
        log_path=str(log),
        git_sha="cafef00d",
        now="2026-07-29T00:00:00+00:00",
    )

    assert result["timestamp"] == "2026-07-29T00:00:00+00:00"
    assert result["git_sha"] == "cafef00d"
    assert result["dataset"] == "unit_b"
    assert result["arm"] == "B"
    assert result["n_documents"] == 2
    assert result["n_gold_clusters"] == 1
    assert result["n_gold_paper_pairs"] == 1
    for strategy in ("cross_product", "same_sentence"):
        assert result["strategies"][strategy]["paper_pair"]["f1"] == 1.0
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1
