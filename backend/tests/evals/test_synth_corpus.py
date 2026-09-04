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


def test_band_for_is_correct_at_every_band_EDGE_not_only_mid_band():
    """Ruling 25. The test above samples 2, 5, 28 and 1 -- all comfortably mid-band -- so
    every boundary was unpinned: widening medium from (4, 7) to (4, 8) moved size 8 out of
    `large` and left all five original tests green. Bands decide which comparison a cluster
    is entered into, so an edge that slips reassigns clusters between the very strata the
    gate reports separately.
    """
    assert [band_for(n) for n in (1, 2, 3)] == [None, "small", "small"]
    assert [band_for(n) for n in (4, 7)] == ["medium", "medium"]
    assert [band_for(n) for n in (8, 9)] == ["large", "large"]
    assert [band_for(n) for n in (10_000, 10_001)] == ["large", None]


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


def test_the_draw_does_not_depend_on_which_same_key_cluster_came_first_in_the_pool():
    """Ruling 25. `sorted(pool, key=lambda c: c.key)` is a STABLE sort, so two clusters
    sharing a key keep their input order through it, and that order feeds `rng.shuffle`.
    `test_the_draw_is_reproducible_from_the_seed_and_not_from_pool_order` claims the draw is
    "not from pool order", but that guarantee held only under an unstated precondition --
    `Cluster.key` uniqueness -- that this module neither checks nor controls. Same multiset,
    same seed, only two same-key clusters swapped: the draw must not change."""
    dup_a = Cluster(key="dup", paper_ids=["a1", "a2"])
    dup_b = Cluster(key="dup", paper_ids=["b1", "b2"])
    rest = _clusters([2] * 18 + [5] * 20 + [9] * 20)

    pool = [dup_a, dup_b, *rest]
    swapped = [dup_b, dup_a, *rest]

    drawn = sample_clusters(pool, rng=random.Random(7), per_band=10)
    drawn_swapped = sample_clusters(swapped, rng=random.Random(7), per_band=10)

    assert [c.paper_ids for c in drawn] == [c.paper_ids for c in drawn_swapped]


def test_sample_clusters_refuses_a_sample_holding_the_same_cluster_twice():
    """Ruling 25, the positive control. It should never fire: Ruling 26 merges clusters by
    key before they reach here, so a repeat means the pooled corpus is corrupt. A duplicate
    double-weights one cluster in a stratified sample of thirty, and `sample_hash` would go
    on producing a perfectly stable hash of the corrupted draw -- the sample would look
    frozen and verified either way, which is why this refuses rather than warns.

    Full identity, not key alone: two clusters may legitimately share a key here (see the
    test above), and it is the SAME cluster arriving twice that is unrecoverable.
    """
    pool = _clusters([2] * 9 + [5] * 10 + [9] * 10)
    duplicate = Cluster(key=pool[0].key, paper_ids=list(pool[0].paper_ids))

    with pytest.raises(ValueError, match="distinct"):
        sample_clusters([*pool, duplicate], rng=random.Random(7), per_band=10)
