import dataclasses

import pytest

from biolit_evals.acronym_export import (
    ACRONYM_LABELS,
    PairEvidence,
    build_rows,
    choose_controls,
    gather_pairs,
    render_markdown,
    rows_hash,
)


def _pair(surface: str = "GSH", **kw) -> PairEvidence:
    base = PairEvidence(
        surface=surface,
        concept_id="MESH:C563177",
        concept_name="Glucocorticoid-Remediable Aldosteronism",
        aliases=("gra", "gsh", "hald1"),
        contexts=("Malondialdehyde (MDA) and glutathione (GSH) were measured.",),
        n_mentions=9,
        type_violation=True,
    )
    return dataclasses.replace(base, **kw)


def test_a_row_carries_the_concept_aliases_because_the_name_alone_can_be_unjudgeable():
    """§3. Two of the 44 concepts are NAMED for the acronym itself -- `APT` -> a concept whose
    canonical name is "APT", `NAD` -> "NAD" -- so a sheet showing only the name would ask the
    annotator to judge a row that contains no information about what the concept is. The
    aliases are what make those rows answerable at all."""
    rows = build_rows([_pair()])

    assert rows[0].aliases == ("gra", "gsh", "hald1")
    assert rows[0].concept_name == "Glucocorticoid-Remediable Aldosteronism"


def test_a_control_keeps_its_own_contexts_and_takes_another_pairs_concept():
    """§4. The control is a REAL surface with its real documents, shown against a REAL concept
    that belongs to a different pair. Both halves are genuine -- nothing here invents a MeSH id
    or an abstract -- and only the pairing is wrong, which is the whole instrument: an
    annotator who marks these `correct` is not reading the text.

    Taking the surface's own contexts matters. A control carrying the *other* pair's contexts
    would be judgeable from the mismatch between surface and text alone, without ever
    consulting the concept."""
    pairs = [
        _pair("GSH"),
        _pair("HCC", concept_id="MESH:D006528", concept_name="Carcinoma, Hepatocellular"),
    ]

    controls = choose_controls(pairs, n=2, seed=7)

    assert len(controls) == 2
    for control in controls:
        source = next(p for p in pairs if p.surface == control.surface)
        assert control.contexts == source.contexts
        assert control.concept_id != source.concept_id


def test_controls_are_not_identifiable_from_their_row_ids():
    """§4 requires the controls be indistinguishable in the export. They are appended after the
    real pairs, so assigning `row_id` before shuffling puts every control in the trailing block
    of ids and an annotator reaching the end of the file would know them on sight.

    This is a defect that actually shipped in the relevance export's first draft, so both
    halves are pinned: the unshuffled call PINS THE HAZARD, so the shuffle cannot be quietly
    dropped and still look tested, and the shuffled call pins that the classes interleave.
    """
    import random

    pairs = [_pair(f"S{i}", concept_id=f"MESH:D{i:05d}") for i in range(20)]
    controls = [_pair(f"C{i}", concept_id=f"MESH:C{i:05d}") for i in range(4)]

    unshuffled = build_rows(pairs, controls=controls)
    assert [i for i, r in enumerate(unshuffled) if r.is_control] == [20, 21, 22, 23], (
        "without a shuffle the controls occupy the trailing ids -- this is the hazard"
    )

    rows = build_rows(pairs, controls=controls, rng=random.Random(7))
    positions = [i for i, r in enumerate(rows) if r.is_control]
    last_real = max(i for i, r in enumerate(rows) if not r.is_control)
    assert len(positions) == 4
    assert min(positions) < last_real, "controls must interleave with the real rows"


def test_the_sheet_hides_the_type_violation_flag_the_mention_count_and_the_controls():
    """§3, and the type-violation half is the one that would quietly invalidate §6.

    If the sheet marked which pairs the linker's own type table already contradicts, those rows
    would be labelled `wrong` on the flag rather than on the text -- and Gate 4 would then be
    reading agreement between the flag and labels the flag itself produced. The flag lives in
    the manifest, which the annotator does not open.

    The mention count is excluded on the relevance pass's argument, which applies here with a
    sharper edge: frequency is not evidence about whether a link is right, and `APT` is
    simultaneously the most frequent pair at 26 mentions and among the clearest errors. Shown,
    it would invite exactly the inference that gets it right for the wrong reason.
    """
    rows = build_rows([_pair(), _pair("HCC", concept_id="MESH:D006528")], controls=[_pair("RA")])
    sheet = render_markdown(rows, rows_hash="deadbeef", seed=1)

    assert "type_violation" not in sheet and "violation" not in sheet.lower()
    assert "9" not in sheet.replace("deadbeef", ""), "mention counts must not reach the sheet"
    assert "control" not in sheet.lower() and "distractor" not in sheet.lower()
    assert sheet.count("label:") == 3, "every row gets the same stub, controls included"
    assert "MESH:" not in sheet, "a raw concept id is a handle into the table being judged"


def test_every_label_the_schema_allows_is_offered_in_the_instructions():
    """A label nobody is told about cannot be used. `cant_tell` is Gate 1's entire instrument,
    and `granularity` is the whole reason the schema is not binary -- if either reads as an
    option that was never offered, a low rate for it means nothing."""
    sheet = render_markdown(build_rows([_pair()]), rows_hash="h", seed=1)

    for label in ACRONYM_LABELS:
        assert f"`{label}`" in sheet


def test_the_sheet_is_readable_by_the_existing_annotation_parser():
    """Reuses `annotation_export.parse_annotations` UNCHANGED for a third pass rather than
    growing a third parser. It refuses a missing or unrecognised label instead of skipping the
    block, which is the property every pass needs: a silently dropped row shrinks a gate
    denominator without changing any visible verdict.

    The context blocks are what make this worth pinning here rather than assuming. They are
    verbatim abstract text inside `>` quotes, so a passage containing a colon, a blank line or
    a stray backtick has to leave the `label:`/`reason:` stub still parseable.
    """
    from biolit_evals.annotation_export import parse_annotations

    awkward = _pair(contexts=("BACKGROUND: levels fell.\n\nRESULTS: `GSH` rose (GSH).",))
    sheet = render_markdown(build_rows([awkward]), rows_hash="h", seed=1).replace(
        "label:\nreason:", "label: wrong\nreason: the abstract says glutathione", 1
    )

    parsed = parse_annotations(sheet, allowed=ACRONYM_LABELS)
    assert parsed["a000"].label == "wrong"
    assert parsed["a000"].reason == "the abstract says glutathione"


_ABSTRACT = (
    "Cisplatin-induced nephrotoxicity in rats. "
    "Malondialdehyde (MDA) and glutathione (GSH) were measured in kidney tissue after "
    "treatment, and GSH depletion tracked the injury."
)


def _state(*entities) -> dict:
    return {
        "candidate_papers": [{"id": "p1", "title": "", "abstract": _ABSTRACT}],
        "extracted_records": {"p1": {"paper_id": "p1", "entities": list(entities)}},
    }


def _entity(text, label, start, cid, cname):
    return {
        "text": text,
        "label": label,
        "start": start,
        "end": start + len(text),
        "canonical_id": cid,
        "canonical_name": cname,
    }


def test_every_context_is_a_verbatim_substring_of_the_source_abstract():
    """⛔ THE RULE THIS PROJECT DOES NOT BEND: never fabricate abstract text. The annotator is
    judging what an author wrote, so a context that paraphrases, re-wraps or repairs the source
    would put the label on a sentence nobody published.

    Windows are cut by offset from the stored abstract and asserted back into it here, rather
    than being reconstructed from the entity's own `text` field -- which is exactly the
    tautological-fixture shape that made an earlier gold loader's validation vacuous.
    """
    gsh = _entity("GSH", "CHEMICAL", _ABSTRACT.index("GSH"), "MESH:C563177", "GRA")
    pairs = gather_pairs([_state(gsh)], concept_labels={"MESH:C563177": ["DISEASE"]})

    assert len(pairs) == 1
    for context in pairs[0].contexts:
        assert context in _ABSTRACT, "a context must be lifted from the source, never rebuilt"
    assert "GSH" in pairs[0].contexts[0]


def test_the_two_contexts_come_from_different_papers_rather_than_twice_from_one():
    """A pair gets at most two windows, and spending both on one abstract wastes the evidence
    budget on the row -- the second mention is usually a few words from the first, so the
    windows overlap and the annotator is shown the same sentence twice.

    It also biases what a `cant_tell` means: a pair whose meaning is settled in a second paper
    would read as unjudgeable purely because both windows were drawn from the first.
    """
    near = _ABSTRACT.index("GSH depletion")
    first = _entity("GSH", "CHEMICAL", _ABSTRACT.index("GSH"), "MESH:C563177", "GRA")
    second = _entity("GSH", "CHEMICAL", near, "MESH:C563177", "GRA")
    other = {
        "candidate_papers": [{"id": "p2", "title": "", "abstract": "Reduced GSH in liver."}],
        "extracted_records": {"p2": {"paper_id": "p2", "entities": [dict(first, start=8, end=11)]}},
    }

    pairs = gather_pairs(
        [_state(first, second), other], concept_labels={"MESH:C563177": ["DISEASE"]}
    )

    assert pairs[0].n_mentions == 3, "every mention still counts toward the weighting"
    assert len(pairs[0].contexts) == 2
    assert any("liver" in context for context in pairs[0].contexts)


def test_the_sheet_caps_the_aliases_and_shows_spelled_out_names_before_acronyms():
    """Some concepts carry 20+ aliases and printing them all buries the row. Two rules, both
    about what the annotator actually needs.

    CAP, so the concept name stays findable rather than ending a wall of text. The full list
    stays in the alias artifact; nothing is lost, it is just not on the sheet.

    SPELLED-OUT FIRST, because an acronym-shaped alias tells the annotator nothing they cannot
    already see in the surface -- `FXS` listing `fxs` among its aliases is noise, while
    `martin-bell syndrome` is the kind of thing that settles a row. Truncating alphabetically
    would keep the noise and drop the signal.
    """
    aliases = (
        "fxs",
        "fra(x) syndrome",
        "abc",
        "martin-bell syndrome",
        *[f"n{i:02d}" for i in range(10)],
    )
    sheet = render_markdown(build_rows([_pair(aliases=aliases)]), rows_hash="h", seed=1)

    listed = sheet.split("**Also known as:** ")[1].split("\n")[0].split(", ")
    assert len(listed) == 8
    assert listed[0] == "fra(x) syndrome" and listed[1] == "martin-bell syndrome"


def test_build_rows_refuses_a_row_that_carries_no_information_about_the_concept():
    """FAIL LOUD. Two of the 44 concepts are named for the acronym itself -- `APT` -> "APT",
    `NAD` -> "NAD" -- and their aliases are the only thing that makes those rows answerable.
    If the alias table were missing or failed to load, the export would emit rows reading
    "does `APT` mean APT?" and the annotator would spend labels on nothing.

    Refusing beats degrading: the failure would otherwise show up as `cant_tell` and get read
    as a Gate 1 tractability problem, when the sheet was simply built wrong.
    """
    with pytest.raises(ValueError, match="APT"):
        build_rows([_pair("APT", concept_name="APT", aliases=())])


def test_rows_hash_tracks_content_not_order():
    """Pins WHICH rows were frozen, so a sheet regenerated after labelling began cannot be
    joined to these labels while claiming the hash they were registered against."""
    # Seven reverse-inserted elements, per this repo's determinism-fixture convention.
    rows = build_rows([_pair(f"S{i}", concept_id=f"MESH:D{i:05d}") for i in range(7)])

    assert rows_hash(rows) == rows_hash(list(reversed(rows)))
    assert rows_hash(rows) != rows_hash(rows[:-1])


def test_controls_avoid_disclosed_surfaces_and_disclosed_concepts_on_both_sides():
    """§7.1. Gate 2 is the instrument that makes every other reading attributable, so a control
    the annotator can reject from MEMORY rather than from the text corrupts the one gate the
    pass cannot afford to lose.

    Both sides are excluded, and the second is the less obvious one. A control on a disclosed
    SURFACE is rejectable because the annotator was told what it means. A control wearing a
    disclosed CONCEPT is rejectable because they were told that concept is a known bogus
    linking target -- recognising `Aphakia, congenital primary` from the `CPA` discussion primes
    the same reflex without a word of the passage being read.
    """
    pairs = [_pair(f"S{i}", concept_id=f"MESH:D{i:05d}", concept_name=f"C{i}") for i in range(12)]
    shown = {"S0", "S1", "S2"}
    shown_concepts = {"MESH:D00003", "MESH:D00004"}

    controls = choose_controls(
        pairs, n=4, seed=7, exclude_surfaces=shown, exclude_concepts=shown_concepts
    )

    assert len(controls) == 4
    for control in controls:
        assert control.surface not in shown
        assert control.concept_id not in shown_concepts
