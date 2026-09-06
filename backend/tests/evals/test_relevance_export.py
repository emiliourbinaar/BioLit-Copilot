import pytest

from biolit_evals.relevance_export import (
    ExportRow,
    build_rows,
    choose_distractors,
    rows_hash,
)


def _names() -> dict[str, str]:
    return {"MESH:C1": "Metformin", "MESH:D1": "Acidosis", "MESH:D2": "Hyperkalemia"}


def test_a_row_shows_concept_names_and_never_the_mesh_ids():
    """§2. The annotator judges topical relevance from concept names. A raw MeSH id is a
    handle into the very table the ranker scores with, and showing it would let an annotator
    reason about identifiers instead of about the question."""
    rows = build_rows(
        [("metformin and lactic acidosis", "MESH:C1|MESH:D1", ["a", "b"], (2001, 2007))],
        names=_names(),
    )

    row = rows[0]
    assert (row.chemical, row.disease) == ("Metformin", "Acidosis")
    assert row.n_papers == 2
    assert row.year_range == "2001-2007"
    assert "MESH:" not in str(row.as_export())
    assert row.as_export()["label"] is None


def test_the_export_row_carries_no_route_back_to_the_cluster_or_the_filter_decision():
    """§3.2/§4. `row_id` must be opaque. If the export leaked the cluster key, the query's
    position, or whether `select_stage` kept it, the labels would measure agreement with a
    decision already seen rather than an independent judgment."""
    rows = build_rows(
        [("metformin and lactic acidosis", "MESH:C1|MESH:D2", ["a"], (2010, 2010))],
        names=_names(),
    )

    exported = rows[0].as_export()
    assert set(exported) == {
        "row_id",
        "query",
        "chemical",
        "disease",
        "n_papers",
        "year_range",
        "label",
    }
    assert exported["row_id"] == "r000"


def test_build_rows_refuses_a_concept_it_cannot_name():
    """FAIL LOUD. Falling back to the id would silently violate §2's no-ids rule on exactly
    the rows where linking is weakest -- and those are the rows most worth judging."""
    with pytest.raises(ValueError, match="no name"):
        build_rows([("q", "MESH:C1|MESH:UNKNOWN", ["a"], (2001, 2001))], names=_names())


def test_a_single_year_renders_without_a_range():
    rows = build_rows([("q", "MESH:C1|MESH:D1", ["a"], (2007, 2007))], names=_names())
    assert rows[0].year_range == "2007"


def test_distractors_come_from_a_different_query_than_the_one_they_are_shown_under():
    """ADR-0018's control. A distractor is only a control if the pairing is genuinely
    unrelated: a cluster shown under its OWN query is a real row, and would make the control
    read as annotator error rather than as discrimination."""
    per_query = {
        "cisplatin nephrotoxicity": ["MESH:C1|MESH:D1"],
        "warfarin and bleeding risk": ["MESH:C2|MESH:D2"],
        "lithium and thyroid dysfunction": ["MESH:C3|MESH:D3"],
    }

    picked = choose_distractors(per_query, n=3, seed=7)

    assert len(picked) == 3
    for shown_under, key in picked:
        assert key not in per_query[shown_under]


def test_the_draw_is_reproducible_from_the_seed():
    per_query = {f"q{i}": [f"MESH:C{i}|MESH:D{i}"] for i in range(8)}

    assert choose_distractors(per_query, n=4, seed=7) == choose_distractors(per_query, n=4, seed=7)
    assert choose_distractors(per_query, n=4, seed=7) != choose_distractors(per_query, n=4, seed=8)


def test_rows_hash_tracks_content_not_order():
    # Seven reverse-inserted elements, per this repo's determinism-fixture convention.
    rows = [
        ExportRow(f"r{i:03d}", f"q{i}", "Chem", "Dis", i, "2001", "MESH:C|MESH:D", False)
        for i in range(7)
    ]

    assert rows_hash(rows) == rows_hash(list(reversed(rows)))
    assert rows_hash(rows) != rows_hash(rows[:-1])


def test_distractors_are_spread_one_per_query_rather_than_massed_in_one():
    """Gate 2 reads the 8 distractors as one verdict about whether the annotator uses
    `off_topic` for its meaning. Massing them in a single question would test discrimination
    only in that question's subject area and report it as a general finding."""
    per_query = {f"q{i}": [f"MESH:C{i}|MESH:D{i}", f"MESH:C{i}|MESH:E{i}"] for i in range(8)}

    picked = choose_distractors(per_query, n=8, seed=20260905)

    assert sorted(shown_under for shown_under, _ in picked) == [f"q{i}" for i in range(8)]


def test_choose_distractors_refuses_when_there_are_too_few_cross_query_pairs():
    with pytest.raises(ValueError, match="need"):
        choose_distractors({"only": ["MESH:C1|MESH:D1"]}, n=8, seed=1)


def test_distractor_rows_are_not_identifiable_from_their_row_ids():
    """§3.4 requires distractors be "indistinguishable in the export". They are appended after
    the real clusters, so assigning `row_id` before shuffling puts every control row in the
    trailing block of ids -- an annotator reaching the end of the file would know them on
    sight and Gate 2 would measure nothing.

    Both halves matter: the unshuffled call PINS THE HAZARD (so the shuffle cannot be quietly
    dropped and still look tested), and the shuffled call pins that the two classes interleave.
    """
    import random

    clusters = [(f"q{i % 4}", "MESH:C1|MESH:D1", ["a", "b"], (2001, 2005)) for i in range(20)]
    distractors = [(f"q{i}", "MESH:C1|MESH:D2") for i in range(4)]

    unshuffled = build_rows(clusters, names=_names(), distractors=distractors)
    assert [i for i, r in enumerate(unshuffled) if r.is_distractor] == [20, 21, 22, 23], (
        "without a shuffle the controls occupy the trailing ids -- this is the hazard"
    )

    rows = build_rows(clusters, names=_names(), distractors=distractors, rng=random.Random(7))
    positions = [i for i, r in enumerate(rows) if r.is_distractor]
    last_real = max(i for i, r in enumerate(rows) if not r.is_distractor)
    assert len(positions) == 4
    assert min(positions) < last_real, "controls must interleave with the real rows"


def test_row_order_is_reproducible_from_the_seed():
    """The frozen row set has to be re-derivable: `rows_hash` pins WHICH rows were shown,
    and this pins the order they were shown in, which the hash deliberately ignores."""
    import random

    clusters = [(f"q{i}", "MESH:C1|MESH:D1", ["a"], (2001, 2001)) for i in range(12)]

    first = build_rows(clusters, names=_names(), rng=random.Random(7))
    again = build_rows(clusters, names=_names(), rng=random.Random(7))
    other = build_rows(clusters, names=_names(), rng=random.Random(8))

    assert [r.query for r in first] == [r.query for r in again]
    assert [r.query for r in first] != [r.query for r in other]
