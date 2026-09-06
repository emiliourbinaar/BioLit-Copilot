from biolit.canon.mesh_tree import MeshTree
from biolit.domain.records import Cluster
from biolit.query.concepts import QueryConcepts
from biolit.query.ranking import rank_clusters

# Synthetic ids throughout. ADR-0020's ranking must NOT be exercised against the eight frozen
# queries anywhere in the test suite: a failure message would print their ordering, and the
# relevance labels for those queries are not yet written (annotation design §8.2).
TREE = MeshTree(
    {
        "MESH:QD": ["C10.100"],
        "MESH:NEAR": ["C10.100.500"],
        "MESH:FAR": ["C10.900"],
        "MESH:UNRELATED": ["F03.200"],
        "MESH:QC": ["D01.100"],
    }
)


def _concepts(*ids: str) -> QueryConcepts:
    return QueryConcepts(frozenset(ids), {i: i for i in ids}, ())


def _cluster(key: str) -> Cluster:
    return Cluster(key=key, paper_ids=["a", "b"])


def test_more_exact_concept_matches_outranks_fewer():
    """Signal 1 dominates. A cluster matching the query on both sides is more on-topic than
    one matching on a single side, whatever the hierarchy says about the other."""
    both = _cluster("MESH:QC|MESH:QD")
    one = _cluster("MESH:QC|MESH:NEAR")

    assert rank_clusters([one, both], _concepts("MESH:QC", "MESH:QD"), tree=TREE) == [
        both,
        one,
    ]


def test_hierarchy_distance_breaks_ties_between_equal_exact_matches():
    """Signal 2, and the reason ADR-0020 needs the tree at all. Both clusters match the query
    chemical and neither matches its disease, so exact-match count cannot separate them --
    but one carries a child of the query's disease concept and the other something from a
    different tree entirely."""
    near = _cluster("MESH:QC|MESH:NEAR")
    unrelated = _cluster("MESH:QC|MESH:UNRELATED")

    ranked = rank_clusters([unrelated, near], _concepts("MESH:QC", "MESH:QD"), tree=TREE)

    assert ranked == [near, unrelated]


def test_a_concept_sharing_no_tree_sorts_below_any_measurable_distance():
    """DEF-0002's failure mode, inverted into a ranking. The disease-required FILTER deleted
    a cluster whose disease side was a sibling of the query's; here the same relationship
    only demotes, and a genuinely unrelated concept goes last rather than being dropped."""
    far = _cluster("MESH:QC|MESH:FAR")
    unrelated = _cluster("MESH:QC|MESH:UNRELATED")

    ranked = rank_clusters([unrelated, far], _concepts("MESH:QC", "MESH:QD"), tree=TREE)

    assert ranked == [far, unrelated]


def test_clusters_the_score_cannot_separate_keep_their_existing_key_order():
    """`cluster_papers` justifies its order as "reproducible and diffable", and ADR-0020
    preserves that as the final tiebreak. Equal relevance must not become an arbitrary
    order -- and per the ADR's extended rule it must not silently acquire a SECOND criterion
    such as size either."""
    first = Cluster(key="MESH:QC|MESH:AAA", paper_ids=["a"])
    second = Cluster(key="MESH:QC|MESH:BBB", paper_ids=["a", "b", "c", "d", "e"])

    assert rank_clusters([second, first], _concepts("MESH:QC"), tree=TREE) == [
        first,
        second,
    ]


def test_ranking_never_adds_or_removes_a_cluster():
    """ADR-0020 orders; it does not select. The measured disaster of using this signal to
    filter -- two of eight answers emptied -- is why the two are kept apart."""
    clusters = [
        _cluster("MESH:QC|MESH:UNRELATED"),
        _cluster("MESH:QC|MESH:QD"),
        _cluster("MESH:ZZZ|MESH:YYY"),
    ]

    ranked = rank_clusters(clusters, _concepts("MESH:QC", "MESH:QD"), tree=TREE)

    assert sorted(c.key for c in ranked) == sorted(c.key for c in clusters)


def test_an_unresolved_query_leaves_the_order_exactly_as_it_was():
    """The fail-open case from `select_stage`, carried through. With no query concepts every
    cluster scores identically, so the result must be the untouched key order rather than an
    incidental reshuffle."""
    clusters = [_cluster("MESH:QC|MESH:AAA"), _cluster("MESH:QC|MESH:BBB")]

    assert (
        rank_clusters(list(reversed(clusters)), QueryConcepts(frozenset(), {}, ()), tree=TREE)
        == clusters
    )


def test_papers_inside_a_cluster_are_not_reordered():
    """ADR-0020 is explicit that the within-cluster no-implicit-ranking rule is untouched.
    Ranking reorders clusters and must never reach inside one."""
    cluster = Cluster(key="MESH:QC|MESH:QD", paper_ids=["z", "a", "m"])

    ranked = rank_clusters([cluster], _concepts("MESH:QC"), tree=TREE)

    assert ranked[0].paper_ids == ["z", "a", "m"]


def test_an_exactly_matched_side_does_not_swamp_the_other_sides_distance():
    """DEF-0003, the regression test. `_relevance_key` minimised over every (side x concept)
    pair, and a cluster's own exactly-matched side gives distance(x, x) == 0, so the minimum
    was 0 for every cluster and the hierarchy term never reached the sort.

    It was TOTAL rather than partial: `select_stage` keeps only clusters with at least one
    exact match, so every cluster the ranker is ever handed had a zero available. Measured on
    the frozen corpus, four of eight queries produced a single distinct score across all their
    clusters and fell back entirely to the MeSH-id order this was built to replace.
    """
    concepts = _concepts("MESH:QC", "MESH:QD")
    near = _cluster("MESH:QC|MESH:NEAR")
    far = _cluster("MESH:QC|MESH:FAR")
    unrelated = _cluster("MESH:QC|MESH:UNRELATED")

    ranked = rank_clusters([unrelated, far, near], concepts, tree=TREE)

    assert ranked == [near, far, unrelated]


def test_proximity_is_the_worst_unmatched_side_not_the_best():
    """A cluster is only as on-topic as its least related side. Taking the best would let one
    good side hide an unrelated one, which is the same shape as the defect above -- a single
    strong signal masking everything else."""
    tree = MeshTree(
        {
            "MESH:QC": ["D01.100"],
            "MESH:QD": ["C10.100"],
            "MESH:BOTHNEAR_A": ["D01.100.500"],
            "MESH:BOTHNEAR_B": ["C10.100.500"],
            "MESH:ONEBAD": ["F03.900"],
        }
    )
    both_near = Cluster(key="MESH:BOTHNEAR_A|MESH:BOTHNEAR_B", paper_ids=["a"])
    one_bad = Cluster(key="MESH:BOTHNEAR_A|MESH:ONEBAD", paper_ids=["a"])

    ranked = rank_clusters([one_bad, both_near], _concepts("MESH:QC", "MESH:QD"), tree=tree)

    assert ranked == [both_near, one_bad]


def test_a_cluster_matching_the_query_on_both_sides_scores_perfect_proximity():
    """No unmatched side means no residual distance. This must not become `inf` by an empty
    max, which would sort the best possible cluster last."""
    concepts = _concepts("MESH:QC", "MESH:QD")
    both = _cluster("MESH:QC|MESH:QD")
    one = _cluster("MESH:QC|MESH:NEAR")

    assert rank_clusters([one, both], concepts, tree=TREE) == [both, one]
