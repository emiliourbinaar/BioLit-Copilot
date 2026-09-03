import random
from pathlib import Path

import pytest

from biolit_evals.alamri_annotation_export import export_stratified_sheet, render_markdown
from biolit_evals.alamri_gold import (
    build_distractors,
    build_pairs,
    parse_corpus,
    question_id,
    read_manifest,
    sample_batch,
    write_manifest,
)
from biolit_evals.annotation_export import parse_annotations

FIXTURE = Path(__file__).parent / "fixtures" / "alamri_fixture.xml"
QUOTAS = {"flagged": 2, "clean": 2, "agreement": 2, "distractor": 1}


@pytest.fixture
def claims():
    return parse_corpus(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def batch(claims):
    pool = build_pairs(claims) + build_distractors(claims)
    return sample_batch(pool, rng=random.Random(7), quotas=QUOTAS, cap_paper=3, cap_question=9)


@pytest.fixture
def abstracts(claims):
    return {c.pmid: f"Abstract text for paper {c.pmid}." for c in claims}


@pytest.fixture
def questions(claims):
    return {question_id(c.question): c.question for c in claims}


def test_every_row_carries_the_same_keys_and_none_of_them_is_the_answer(
    batch, abstracts, questions
):
    """The blind protocol's core invariant. A key present on some strata and absent on
    others is a structural tell even if its value is innocuous, so the key SET is asserted
    rather than merely the absence of `label`."""
    rows = export_stratified_sheet(batch, abstracts, questions, rng=random.Random(3))
    assert {frozenset(row) for row in rows} == {
        frozenset({"pair_id", "question", "abstract_a", "abstract_b"})
    }


def test_a_distractor_row_shows_exactly_one_of_its_two_questions(claims, abstracts, questions):
    """A two-question row identifies a distractor with certainty. Showing one turns it into
    an honest instance of the same task every other row poses, whose correct answer happens
    to be insufficient_overlap because the second paper does not address the question."""
    distractors = build_distractors(claims)
    batch = sample_batch(
        distractors, rng=random.Random(7), quotas={"distractor": 3}, cap_paper=3, cap_question=9
    )
    by_id = {p.pair_id: p for p in batch}
    rows = export_stratified_sheet(batch, abstracts, questions, rng=random.Random(3))
    assert rows
    for row in rows:
        pair = by_id[row["pair_id"]]
        shown = {question_id(row["question"])}
        assert shown < {pair.question_id_a, pair.question_id_b}


def test_the_two_abstracts_are_not_always_presented_in_manifest_order(batch, abstracts, questions):
    """In the manifest, `paper_id_a` of a contradiction pair is always the YS paper. Left
    unshuffled, "abstract A answers yes" would hold across every contradiction row in the
    batch -- a regularity that says nothing about whether the two papers disagree."""
    by_id = {p.pair_id: p for p in batch}
    swapped = 0
    for seed in range(12):
        for row in export_stratified_sheet(batch, abstracts, questions, rng=random.Random(seed)):
            pair = by_id[row["pair_id"]]
            if row["abstract_a"] == abstracts[pair.paper_id_b]:
                swapped += 1
    assert swapped > 0


def test_rendered_markdown_round_trips_through_the_reused_parser(batch, abstracts, questions):
    """`parse_annotations` is reused UNMODIFIED from Phase 5, so this sheet must match the
    block format it already reads. If it does not, the gates read a batch with rows missing
    and a dropped `contradiction` lowers g -- pushing Gate 2 toward the verdict that retires
    the design, on a formatting bug."""
    rows = export_stratified_sheet(batch, abstracts, questions, rng=random.Random(3))
    sheet = render_markdown(rows, corpus_hash="deadbeef", seed=3)
    filled = sheet.replace("label:\n", "label: contradiction\n").replace(
        "reason:\n", "reason: because\n"
    )
    parsed = parse_annotations(filled)
    assert set(parsed) == {row["pair_id"] for row in rows}
    assert all(a.label == "contradiction" for a in parsed.values())


def test_rows_are_not_left_grouped_by_stratum(batch, abstracts, questions):
    """`sample_batch` returns strata in draw order, so an unshuffled sheet would present
    every flagged pair, then every clean pair, then the controls -- position alone would
    identify the class."""
    by_id = {p.pair_id: p for p in batch}
    strata_in_draw_order = [p.stratum for p in batch]
    orders = {
        tuple(
            by_id[row["pair_id"]].stratum
            for row in export_stratified_sheet(batch, abstracts, questions, rng=random.Random(seed))
        )
        for seed in range(12)
    }
    assert orders != {tuple(strata_in_draw_order)}


def test_rendering_the_sheet_does_not_disturb_the_frozen_manifest(
    batch, abstracts, questions, tmp_path
):
    """The display swap is render-time ONLY. `paper_id_a` of a contradiction pair is the YS
    paper in the manifest and must stay so however the sheet presents it -- the manifest is
    what a later reader re-derives the gold from, and a swap leaking into it would silently
    invert which paper answered yes."""
    path = tmp_path / "batch.jsonl"
    write_manifest(batch, path)
    before = path.read_bytes()

    rows = export_stratified_sheet(batch, abstracts, questions, rng=random.Random(3))
    write_manifest(batch, path)

    by_id = {p.pair_id: p for p in read_manifest(path)}
    displayed_swapped = [
        r for r in rows if r["abstract_a"] == abstracts[by_id[r["pair_id"]].paper_id_b]
    ]
    assert displayed_swapped, "seed produced no swapped row; the invariant is untested"
    assert path.read_bytes() == before
    assert all(p == q for p, q in zip(batch, [by_id[p.pair_id] for p in batch], strict=True))
