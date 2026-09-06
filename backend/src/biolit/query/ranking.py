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

from biolit.canon.mesh_tree import MeshTree
from biolit.domain.records import Cluster
from biolit.query.concepts import QueryConcepts

#: Sorts after every real distance. `MeshTree.distance` returns None for "no shared placement",
#: which is a category rather than a magnitude, so it is mapped here at the point of sorting
#: instead of being invented inside the tree.
_NO_SHARED_TREE = float("inf")


def _relevance_key(cluster: Cluster, concepts: QueryConcepts, tree: MeshTree) -> tuple:
    """Lexicographic: exact matches (more first), then hierarchy proximity, then key order.

    No threshold anywhere. Each signal is ordinal and the sort consumes it as such, so there
    is no constant to tune and none can be tuned against the relevance labels later.
    """
    sides = cluster.key.split("|")
    exact = sum(1 for side in sides if side in concepts.ids)

    distances = [
        distance
        for side in sides
        for concept_id in concepts.ids
        if (distance := tree.distance(side, concept_id)) is not None
    ]
    proximity = min(distances) if distances else _NO_SHARED_TREE

    # `cluster.key` last preserves `cluster_papers`'s reproducible-and-diffable guarantee for
    # clusters the score cannot separate.
    return (-exact, proximity, cluster.key)


def rank_clusters(
    clusters: Sequence[Cluster], concepts: QueryConcepts, *, tree: MeshTree
) -> list[Cluster]:
    """Return the clusters ordered by relevance. Adds nothing, removes nothing.

    ⛔ THIS SIGNAL MUST NOT BE USED TO FILTER. Measured on the frozen corpus, requiring a
    hierarchy match instead of merely preferring one keeps 33 of 83 clusters and returns an
    EMPTY answer on 2 of 8 queries -- because a query resolving to `Depressive Disorder` finds
    clusters carrying `Mental Disorders` and `Anxiety Disorders`, which are its parent and
    siblings rather than itself. Here that same relationship only demotes. A ranking error
    moves a cluster down the page; a filter error deletes it.
    """
    return sorted(clusters, key=lambda cluster: _relevance_key(cluster, concepts, tree))
