import json
import random

import pytest

from biolit.domain.records import ContradictionLabel
from biolit_evals.contradiction_gold import (
    GoldPair,
    assert_labels_rederive,
    assert_no_bc5cdr_pmids,
    assert_one_pair_per_key,
    assert_papers_disjoint,
    build_candidates,
    label_for_directions,
    manifest_hash,
    read_manifest,
    sample_pairs,
    write_manifest,
)
from biolit_evals.ctd_directions import Direction

_MM = frozenset({Direction.marker_mechanism})
_TH = frozenset({Direction.therapeutic})


def test_opposite_directions_are_a_contradiction():
    assert label_for_directions(_MM, _TH) is ContradictionLabel.contradiction
    assert label_for_directions(_TH, _MM) is ContradictionLabel.contradiction


def test_same_direction_is_agreement():
    assert label_for_directions(_MM, _MM) is ContradictionLabel.agreement
    assert label_for_directions(_TH, _TH) is ContradictionLabel.agreement


def test_paper_with_both_directions_on_one_key_is_not_gold():
    assert label_for_directions(_MM | _TH, _TH) is None


def test_both_directions_on_the_SECOND_paper_is_also_not_gold():
    """The second disjunct. Without it a b-side double-direction pair is labelled
    `contradiction` because the frozensets merely differ."""
    assert label_for_directions(_MM, _MM | _TH) is None


def test_excluded_pmid_drops_its_pairs():
    """p2 is a BC5CDR training pmid. Without the exclusion it would pair with p1 and p3
    (all three share key C1/D1); with it, no candidate may name p2."""
    directions = {
        "p1": {("C1", "D1"): _MM},
        "p2": {("C1", "D1"): _MM},
        "p3": {("C1", "D1"): _MM},
    }
    candidates = build_candidates(directions, excluded={"p2"})
    named_papers = {pid for c in candidates for pid in (c.paper_id_a, c.paper_id_b)}
    assert "p2" not in named_papers
    assert named_papers == {"p1", "p3"}


def test_curated_key_pairs_get_both_endpoint_ids_set():
    """A co-keyed pair with agreeing directions and one with opposing directions must both
    carry the full (chemical, disease) key -- that is what distinguishes them from a hard
    negative, which has exactly one endpoint set."""
    directions = {
        "p1": {("C1", "D1"): _MM},
        "p2": {("C1", "D1"): _MM},
        "p3": {("C2", "D2"): _MM},
        "p4": {("C2", "D2"): _TH},
    }
    candidates = build_candidates(directions, excluded=set())
    agreement = next(c for c in candidates if c.label is ContradictionLabel.agreement)
    contradiction = next(c for c in candidates if c.label is ContradictionLabel.contradiction)
    assert (agreement.chemical_id, agreement.disease_id) == ("C1", "D1")
    assert (contradiction.chemical_id, contradiction.disease_id) == ("C2", "D2")


def test_hard_negative_has_exactly_one_endpoint_id_and_no_directions():
    """p1 and p2 share chemical C1 (via different diseases) but no curated key joins them --
    that is the definition of insufficient_overlap. Exactly one endpoint id is set."""
    directions = {
        "p1": {("C1", "D1"): _MM},
        "p2": {("C1", "D2"): _TH},
    }
    candidates = build_candidates(directions, excluded=set())
    assert len(candidates) == 1
    negative = candidates[0]
    assert negative.label is ContradictionLabel.insufficient_overlap
    assert (negative.chemical_id is None) != (negative.disease_id is None)
    assert negative.chemical_id == "C1"
    assert negative.direction_a is None
    assert negative.direction_b is None


def test_max_per_key_caps_emission():
    """8 papers on one key, all agreeing, form 28 combinations -- real CTD has ~2.78M such
    pairs on its most popular keys. Only max_per_key may be materialised."""
    directions = {f"p{i}": {("C1", "D1"): _MM} for i in range(8, 0, -1)}
    candidates = build_candidates(directions, excluded=set(), max_per_key=3)
    assert len(candidates) == 3
    assert all(c.label is ContradictionLabel.agreement for c in candidates)


def test_build_candidates_is_deterministic():
    directions = {
        f"p{i}": {("C1", "D1"): _MM} if i % 2 else {("C1", "D1"): _TH} for i in range(7, 0, -1)
    }
    first = build_candidates(directions, excluded=set())
    second = build_candidates(directions, excluded=set())
    assert first == second


def test_build_candidates_output_is_in_ascending_lexicographic_order():
    """Insertion order is adversarial (descending) at both the key level and the pmid-within-
    key level, so this only passes if `sorted()` is actually doing the ordering -- not because
    the input happened to already be sorted. Canonical order is what makes the seeded shuffle
    in `sample_pairs` reproducible across processes and hash seeds."""
    directions = {
        "p9": {("C2", "D2"): _MM},
        "p8": {("C2", "D2"): _MM},
        "p4": {("C1", "D1"): _MM},
        "p3": {("C1", "D1"): _MM},
    }
    candidates = build_candidates(directions, excluded=set())
    assert len(candidates) == 2
    assert (candidates[0].chemical_id, candidates[0].disease_id) == ("C1", "D1")
    assert (candidates[0].paper_id_a, candidates[0].paper_id_b) == ("p3", "p4")
    assert (candidates[1].chemical_id, candidates[1].disease_id) == ("C2", "D2")
    assert (candidates[1].paper_id_a, candidates[1].paper_id_b) == ("p8", "p9")


def test_no_paper_appears_in_two_sampled_pairs():
    """Load-bearing for the statistics: shared papers make the trials dependent, and every
    binomial interval in the report would then be understated."""
    candidates = [
        GoldPair(
            f"p{i}",
            "SHARED",
            f"C{i:06d}",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
        for i in range(7, 0, -1)  # 7 elements, reverse-inserted
    ]
    sampled = sample_pairs(candidates, per_class=7, rng=random.Random(0))
    assert len(sampled) == 1


def test_only_one_pair_per_key_is_drawn():
    candidates = [
        GoldPair(
            f"a{i}",
            f"b{i}",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
        for i in range(7, 0, -1)
    ]
    assert len(sample_pairs(candidates, per_class=7, rng=random.Random(0))) == 1


def test_reuse_of_paper_b_alone_is_also_rejected():
    """`a in used or b in used` -- the b-side operand. Without it, a candidate reusing only
    its second paper is admitted and the trials stop being independent."""
    candidates = [
        GoldPair(
            "a1",
            "shared",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
        GoldPair(
            "a2",
            "shared",
            "C000002",
            "D000002",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
    ]
    assert len(sample_pairs(candidates, per_class=2, rng=random.Random(0))) == 1


def test_reuse_of_paper_a_alone_is_also_rejected():
    """`a in used or b in used` -- the a-side operand. Without it, a candidate reusing only
    its first paper is admitted and the trials stop being independent. The brief's own fixture
    for `test_no_paper_appears_in_two_sampled_pairs` only shares paper_id_b across candidates,
    so it cannot exercise this operand; this test mirrors it on the a-side."""
    candidates = [
        GoldPair(
            "shared",
            "b1",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
        GoldPair(
            "shared",
            "b2",
            "C000002",
            "D000002",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
    ]
    assert len(sample_pairs(candidates, per_class=2, rng=random.Random(0))) == 1


def test_sampling_is_deterministic_given_a_seed():
    candidates = [
        GoldPair(
            f"a{i}",
            f"b{i}",
            f"C{i:06d}",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
        for i in range(7, 0, -1)
    ]
    first = sample_pairs(candidates, per_class=7, rng=random.Random(11))
    second = sample_pairs(candidates, per_class=7, rng=random.Random(11))
    assert [p.paper_id_a for p in first] == [p.paper_id_a for p in second]
    assert [p.paper_id_a for p in first] != [p.paper_id_a for p in candidates]


def test_manifest_round_trips(tmp_path):
    pairs = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    path = tmp_path / "m.jsonl"
    write_manifest(pairs, path)
    assert read_manifest(path) == pairs


def test_manifest_round_trip_preserves_enum_identity_not_just_equality(tmp_path):
    """`==` is True for a plain str because ContradictionLabel is a StrEnum, so an equality
    round-trip test passes while `is` comparisons break. assert_labels_rederive uses `is`."""
    pairs = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    path = tmp_path / "m.jsonl"
    write_manifest(pairs, path)
    loaded = read_manifest(path)
    assert loaded[0].label is ContradictionLabel.contradiction


def test_labels_rederive_anchor_fires_on_a_mislabelled_pair():
    bad = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.agreement,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    with pytest.raises(AssertionError, match="re-derive"):
        assert_labels_rederive(bad)


def test_labels_rederive_anchor_fires_when_direction_a_is_missing():
    """`direction_a is None or direction_b is None` -- the a-side operand. A pair labelled
    contradiction/agreement must carry both directions; a missing a-side must fire on its
    own, not merely when b is also missing."""
    bad = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            None,
            "therapeutic",
        )
    ]
    with pytest.raises(AssertionError, match="no recorded directions"):
        assert_labels_rederive(bad)


def test_manifest_hash_is_stable_over_content_not_file_bytes(tmp_path):
    """manifest_hash is documented to be stable over CONTENT, not file bytes -- a run-log
    line pins a corpus by this hash, and CTD is a living database, so a re-serialization
    (or writing the same content to a different path) must never change it. Written to two
    different paths and reloaded, the same logical content must hash identically both to
    each other and to the original in-memory list."""
    pairs = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        )
    ]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "nested" / "b.jsonl"
    write_manifest(pairs, path_a)
    write_manifest(pairs, path_b)
    assert manifest_hash(pairs) == manifest_hash(read_manifest(path_a))
    assert manifest_hash(read_manifest(path_a)) == manifest_hash(read_manifest(path_b))

    # Same logical content, hand-written with a different raw key order -- if the hash ever
    # depended on the file's actual bytes rather than the re-serialized loaded content,
    # this would diverge from the two above.
    reordered_path = tmp_path / "reordered.jsonl"
    reordered_line = json.dumps(
        {
            "direction_b": "therapeutic",
            "label": "contradiction",
            "paper_id_a": "11",
            "chemical_id": "C000001",
            "disease_id": "D000001",
            "paper_id_b": "22",
            "direction_a": "marker/mechanism",
        }
    )
    reordered_path.write_text(reordered_line + "\n", encoding="utf-8")
    assert manifest_hash(read_manifest(reordered_path)) == manifest_hash(pairs)


def test_disjointness_anchor_fires_on_a_reused_paper():
    pairs = [
        GoldPair(
            "a",
            "shared",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
        GoldPair(
            "b",
            "shared",
            "C000002",
            "D000002",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
    ]
    with pytest.raises(AssertionError, match="appears in 2 pairs"):
        assert_papers_disjoint(pairs)


def test_bc5cdr_anchor_fires_on_an_excluded_pmid():
    pairs = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
    ]
    with pytest.raises(AssertionError, match="BC5CDR"):
        assert_no_bc5cdr_pmids(pairs, excluded={"11"})


def test_one_pair_per_key_anchor_fires_on_a_reused_key():
    pairs = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
        GoldPair(
            "33",
            "44",
            "C000001",
            "D000001",
            ContradictionLabel.agreement,
            "marker/mechanism",
            "marker/mechanism",
        ),
    ]
    with pytest.raises(AssertionError, match="contributes 2 pairs"):
        assert_one_pair_per_key(pairs)


def test_labels_rederive_passes_silently_on_a_valid_insufficient_overlap_pair():
    """Positive control for the `insufficient_overlap` skip. The real corpus always contains
    such pairs (both directions are None BY CONSTRUCTION -- see build_candidates), so if the
    skip ever regressed, this anchor would fire on every perfectly healthy corpus. A positive
    control that cries wolf is worse than none: the first response would be to distrust the
    corpus rather than the anchor."""
    pairs = [
        GoldPair(
            "11",
            "22",
            "C000001",
            None,
            ContradictionLabel.insufficient_overlap,
            None,
            None,
        ),
        GoldPair(
            "33",
            "44",
            "C000002",
            "D000002",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            "therapeutic",
        ),
    ]
    assert_labels_rederive(pairs)  # must not raise


def test_labels_rederive_anchor_fires_when_direction_b_is_missing():
    """`direction_a is None or direction_b is None` -- the b-side operand. Without it a pair
    missing only its second direction is admitted and label re-derivation is skipped
    silently."""
    bad = [
        GoldPair(
            "11",
            "22",
            "C000001",
            "D000001",
            ContradictionLabel.contradiction,
            "marker/mechanism",
            None,
        )
    ]
    with pytest.raises(AssertionError, match="no recorded directions"):
        assert_labels_rederive(bad)
