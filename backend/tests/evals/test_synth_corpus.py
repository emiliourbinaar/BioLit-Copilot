import random

import pytest

from biolit.domain.records import Cluster
from biolit_evals.synth_corpus import band_for, sample_clusters, sample_hash


def _clusters(sizes: list[int]) -> list[Cluster]:
    return [
        Cluster(key=f"k{i}", paper_ids=[f"p{i}_{j}" for j in range(size)])
        for i, size in enumerate(sizes)
    ]


def test_band_for_maps_sizes_to_the_three_bands():
    assert band_for(2) == "small"
    assert band_for(5) == "medium"
    assert band_for(28) == "large"
    assert band_for(1) is None


def test_sample_clusters_draws_the_quota_from_every_band():
    """Stratification is required by the measured size skew, not a preference: top-5
    clusters carry 50.3% of comparisons, so an unstratified draw is nearly all small ones."""
    pool = _clusters([2] * 20 + [5] * 20 + [9] * 20)

    got = sample_clusters(pool, rng=random.Random(7), per_band=10)

    assert len(got) == 30
    counts = {
        b: sum(1 for c in got if band_for(len(c.paper_ids)) == b)
        for b in ("small", "medium", "large")
    }
    assert counts == {"small": 10, "medium": 10, "large": 10}


def test_sample_clusters_refuses_a_band_it_cannot_fill():
    """A short band would silently change what the gate measures -- the same reason
    `sample_batch` refuses a short stratum in the Alamri harness."""
    pool = _clusters([2] * 20 + [5] * 20 + [9] * 3)

    with pytest.raises(ValueError, match="large"):
        sample_clusters(pool, rng=random.Random(7), per_band=10)


def test_sample_hash_tracks_content_not_order():
    # Seven reverse-inserted elements, per this repo's determinism-fixture convention: with
    # two or three, a hash that accidentally depended on order could still pass by luck.
    a = _clusters([2, 3, 4, 5, 7, 9, 12])
    b = list(reversed(a))

    assert sample_hash(a) == sample_hash(b)
    assert sample_hash(a) != sample_hash(_clusters([2, 3, 4, 5, 7, 9, 11]))


def test_the_draw_is_reproducible_from_the_seed_and_not_from_pool_order():
    """Ruling 21. The manifest hash pins WHICH clusters were scored; this pins that the
    draw behind it can be re-derived. The reversed-pool half is not redundant -- it pins the
    `sorted()` before the shuffle, without which the same seed draws a different sample from
    a differently-ordered pipeline dump, and dump order is not something the manifest
    records."""
    pool = _clusters([2] * 20 + [5] * 20 + [9] * 20)

    same_seed = sample_clusters(pool, rng=random.Random(7), per_band=10)
    reversed_pool = sample_clusters(list(reversed(pool)), rng=random.Random(7), per_band=10)
    other_seed = sample_clusters(pool, rng=random.Random(8), per_band=10)

    assert [c.key for c in same_seed] == [c.key for c in reversed_pool]
    assert [c.key for c in same_seed] != [c.key for c in other_seed]
