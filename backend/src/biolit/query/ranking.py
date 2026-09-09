"""Order clusters by relevance to the stated query (ADR-0020).

WHAT THIS DOES AND DOES NOT CLAIM. Cluster order carries exactly one meaning -- relevance to
the question the reader asked -- and no other. Ranking a cluster higher says nothing about its
papers being better, more reliable, more numerous or more important; only that they are nearer
the question. Where relevance does not distinguish two clusters the order stays deterministic
and encodes no second criterion.

⚠️ SIZE IS DELIBERATELY NOT A SIGNAL. It is the obvious tiebreaker and it is precisely the
manufactured importance hierarchy that `render_cluster`'s within-cluster rule forbids. It is
also what currently makes `Isotretinoin | Acne Vulgaris` look authoritative in the answer to a
depression question: it is the largest cluster on that query and the least on-query.

The within-cluster ordering of papers is untouched. That is where the "no implicit ranking"
rule actually lives, and this module never reaches inside a cluster.
"""

from collections.abc import Sequence

from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.canon.mesh_tree import MeshTree
from biolit.domain.records import Cluster
from biolit.query.concepts import QueryConcepts, side_matches

#: Sorts after every real distance. `MeshTree.distance` returns None for "no shared placement",
#: which is a category rather than a magnitude, so it is mapped here at the point of sorting
#: instead of being invented inside the tree.
_NO_SHARED_TREE = float("inf")


def relevance_score(
    cluster: Cluster,
    concepts: QueryConcepts,
    *,
    tree: MeshTree,
    actions: PharmacologicalActions,
) -> tuple[int, float]:
    """The two ordinal signals behind the sort, without the sort's negation.

    Exposed so a consumer can DISPLAY why a cluster ranks where it does. It is deliberately
    the same computation `_relevance_key` consumes rather than a parallel one: a second
    implementation of a score is a second thing to keep in step, and this project has already
    recorded what happens when a displayed number and a computed number drift apart.
    """
    sides = cluster.key.split("|")
    matched = sum(1 for side in sides if side_matches(side, concepts, actions))

    # RESIDUAL match quality: how far are the sides that are NOT already exact matches?
    #
    # ⚠️ DEF-0003. This was once a `min` over every (side x concept) pair, and that made the
    # whole term dead: a cluster's exactly-matched side gives `distance(x, x) == 0`, so the
    # minimum was 0 for every cluster. It was total rather than partial, because `select_stage`
    # only ever hands over clusters with at least one MATCHED side. Four of eight frozen queries
    # scored a single distinct value across all their clusters and fell back entirely to the
    # MeSH-id order this exists to replace.
    #
    # ADR-0022 widened what "matched" means and the exclusion below widened with it, so the
    # invariant is preserved rather than merely still true by luck. It has to be: a class
    # member shares NO tree node with its class -- Atorvastatin sits in D03/D10, the statin
    # class in D27 -- so a side matched by the class relation and left in the residual scores
    # `inf` and sorts last. The wider filter would have made the ranker worse, not just unfixed.
    #
    # MAX over the unmatched sides, not min: a cluster is only as on-topic as its LEAST related
    # side, and taking the best would let one strong side hide an unrelated one -- the same
    # masking shape as the defect itself. An empty max means every side matched exactly, which
    # is the best possible cluster and scores 0 rather than infinity.
    residual = [
        min(
            (
                distance
                for concept_id in concepts.ids
                if (distance := tree.distance(side, concept_id)) is not None
            ),
            default=_NO_SHARED_TREE,
        )
        for side in sides
        if not side_matches(side, concepts, actions)
    ]
    return matched, max(residual, default=0.0)


def _relevance_key(
    cluster: Cluster, concepts: QueryConcepts, tree: MeshTree, actions: PharmacologicalActions
) -> tuple:
    """Lexicographic: matched sides (more first), then hierarchy proximity, then key order.

    No threshold anywhere. Each signal is ordinal and the sort consumes it as such, so there
    is no constant to tune and none can be tuned against the relevance labels later. ADR-0022
    keeps that property: `side_matches` is a set relation, not a distance with a cutoff.

    `cluster.key` last preserves `cluster_papers`'s reproducible-and-diffable guarantee for
    clusters the score cannot separate.
    """
    matched, proximity = relevance_score(cluster, concepts, tree=tree, actions=actions)
    return (-matched, proximity, cluster.key)


def rank_clusters(
    clusters: Sequence[Cluster],
    concepts: QueryConcepts,
    *,
    tree: MeshTree,
    actions: PharmacologicalActions,
) -> list[Cluster]:
    """Return the clusters ordered by relevance. Adds nothing, removes nothing.

    ⛔ THIS SIGNAL MUST NOT BE USED TO FILTER. Measured on the frozen corpus, requiring a
    hierarchy match instead of merely preferring one keeps 33 of 83 clusters and returns an
    EMPTY answer on 2 of 8 queries -- because a query resolving to `Depressive Disorder` finds
    clusters carrying `Mental Disorders` and `Anxiety Disorders`, which are its parent and
    siblings rather than itself. Here that same relationship only demotes. A ranking error
    moves a cluster down the page; a filter error deletes it.
    """
    return sorted(clusters, key=lambda cluster: _relevance_key(cluster, concepts, tree, actions))
