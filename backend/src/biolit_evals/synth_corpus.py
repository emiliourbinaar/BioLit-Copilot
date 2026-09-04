"""Freeze real pipeline clusters and draw the size-stratified Gate A sample.

Sampling is stratified because the size skew is extreme and measured: `top5_pair_share` on
Arm B `same_sentence` is 0.503 -- half of all cluster comparisons come from five clusters --
the largest Arm B cluster holds 11 papers and the largest Arm A cluster 28. An unstratified
draw would be almost entirely 2-paper clusters, and a 2-paper cluster is a different task
from a 12-paper one.
"""

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Sequence

from biolit.domain.records import Cluster

#: (name, min_papers, max_papers), inclusive. A cluster of 1 is not a cluster.
SIZE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("small", 2, 3),
    ("medium", 4, 7),
    ("large", 8, 10_000),
)


def band_for(size: int) -> str | None:
    """The band a cluster of `size` papers belongs to, or None if it is unusable."""
    for name, low, high in SIZE_BANDS:
        if low <= size <= high:
            return name
    return None


def sample_clusters(
    clusters: Sequence[Cluster], *, rng: random.Random, per_band: int = 10
) -> list[Cluster]:
    """Draw `per_band` clusters from each size band, in band order.

    RAISES on a band it cannot fill. A short band silently changes what the gate measures --
    the same failure the Alamri sampler refuses for the same reason -- and a per-band metric
    computed over four clusters instead of ten would be reported as though it were the
    designed comparison.
    """
    by_band: dict[str, list[Cluster]] = defaultdict(list)
    for cluster in clusters:
        band = band_for(len(cluster.paper_ids))
        if band is not None:
            by_band[band].append(cluster)

    drawn: list[Cluster] = []
    for name, _low, _high in SIZE_BANDS:
        pool = sorted(by_band.get(name, []), key=lambda c: c.key)
        if len(pool) < per_band:
            raise ValueError(
                f"sample_clusters: band {name!r} holds {len(pool)} clusters, need {per_band}. "
                "Widen the query set rather than shrinking the quota -- a short band would "
                "change what the gate measures without saying so."
            )
        rng.shuffle(pool)
        drawn.extend(pool[:per_band])
    return drawn


def sample_hash(clusters: Sequence[Cluster]) -> str:
    """Stable over content, not over order or file bytes.

    Sorted before hashing so a re-ordered draw of the same clusters hashes identically: the
    hash answers "which clusters were scored", and draw order is already pinned by the seed.
    Same posture as `contradiction_gold.manifest_hash` -- content, never serialization.
    """
    payload = json.dumps(
        sorted((c.key, sorted(c.paper_ids)) for c in clusters), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
