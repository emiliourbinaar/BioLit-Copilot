from biolit.ner.windowing import plan_windows, predict_windowed


def _chars(text: str) -> int:
    """Fake token counter: 1 'token' per character, so budgets are easy to reason about."""
    return len(text)


def test_short_text_is_a_single_window_covering_everything():
    text = "Metformin treats PCOS."
    windows = plan_windows(text, count_tokens=_chars, max_tokens=1000)
    assert windows == [(0, len(text))]


def test_every_window_fits_the_budget_and_all_text_is_covered():
    # Ten sentences of ~30 chars each with a 100-"token" budget forces several windows.
    text = " ".join(f"Sentence number {i} mentions metformin." for i in range(10))
    windows = plan_windows(text, count_tokens=_chars, max_tokens=100)
    assert len(windows) > 1
    for start, end in windows:
        assert _chars(text[start:end]) <= 100
    # Union of windows must cover every character: no gap can silently drop content.
    covered = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(len(text)))


def test_no_window_boundary_is_a_hard_cut():
    # The straddle guarantee, stated as the property that actually protects entities: every
    # internal window boundary is spanned by some OTHER window, so text sitting across a
    # boundary is still seen intact by the model at least once. Without overlap this fails
    # -- each cut point would be a hard edge, truncating anything crossing it exactly the
    # way the original 512-token overflow truncated the tail of a document.
    text = " ".join(f"Sentence number {i} mentions metformin." for i in range(10))
    windows = plan_windows(text, count_tokens=_chars, max_tokens=100)
    boundaries = [end for _, end in windows[:-1]]
    assert boundaries, windows
    for boundary in boundaries:
        assert any(start < boundary < end for start, end in windows), (boundary, windows)


def test_without_overlap_boundaries_are_hard_cuts():
    # Pins that the previous test is not vacuous: the same text planned with no overlap
    # leaves at least one boundary no window spans.
    text = " ".join(f"Sentence number {i} mentions metformin." for i in range(10))
    windows = plan_windows(text, count_tokens=_chars, max_tokens=100, overlap_sentences=0)
    boundaries = [end for _, end in windows[:-1]]
    assert boundaries, windows
    assert not all(any(s < b < e for s, e in windows) for b in boundaries)


def test_single_sentence_longer_than_budget_is_split_rather_than_dropped():
    # Pathological but must not crash or lose text: one sentence exceeding the whole budget.
    text = "word " * 100
    windows = plan_windows(text, count_tokens=_chars, max_tokens=60)
    assert len(windows) > 1
    for start, end in windows:
        assert _chars(text[start:end]) <= 60
    covered: set[int] = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(len(text)))


def test_unbroken_run_longer_than_budget_still_yields_windows_that_fit():
    # A whitespace-free run longer than the budget has no split point, so the greedy
    # word-level fallback used to emit it whole -- handing the model an over-budget window
    # and re-raising the very tensor-size error this module exists to prevent.
    text = "A" * 300 + " short tail here."
    windows = plan_windows(text, count_tokens=_chars, max_tokens=50)
    assert windows, text
    for start, end in windows:
        assert _chars(text[start:end]) <= 50, (start, end, windows)
    covered: set[int] = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(len(text)))


def test_empty_text_plans_no_windows():
    assert plan_windows("", count_tokens=_chars, max_tokens=100) == []


def test_a_span_contained_in_another_is_dropped_but_distinct_repeats_survive():
    # Overlapping windows can see the same entity twice at different extents -- once whole,
    # once clipped by a window edge. Keying only on (start, end, group) let BOTH survive,
    # emitting nested spans that never occur on single-window input and that
    # merge_fragments then treats as connector-joined (a zero-length gap counts as a
    # connector). Keep the longest and drop the contained one. Two SEPARATE occurrences of
    # the same surface at different offsets are not contained in each other and must both
    # survive.
    text = " ".join(f"Sentence {i} gave sodium chloride to patients." for i in range(8))
    whole, clipped = "sodium chloride", "sodium"

    def fake_run(chunk: str) -> list[dict]:
        spans = []
        for probe in (whole, clipped):
            cursor = chunk.find(probe)
            while cursor != -1:
                spans.append(
                    {
                        "entity_group": "Chemical",
                        "score": 0.9,
                        "word": probe,
                        "start": cursor,
                        "end": cursor + len(probe),
                    }
                )
                cursor = chunk.find(probe, cursor + 1)
        return spans

    spans = predict_windowed(text, run=fake_run, count_tokens=_chars, max_tokens=200)
    kept = [(s["start"], s["end"]) for s in spans]
    # Every clipped "sodium" is contained in a "sodium chloride" and must be dropped...
    assert all(text[s:e] == whole for s, e in kept), kept
    # ...while each of the 8 distinct occurrences survives on its own offsets.
    assert len(kept) == text.count(whole), kept
    assert len({s for s, _ in kept}) == len(kept)


def test_predict_windowed_shifts_offsets_to_document_coordinates_and_dedupes():
    # A fake "pipeline" that finds every occurrence of "metformin" in whatever window text
    # it is handed, reporting WINDOW-relative offsets -- exactly what the real pipeline
    # does. The word appears in several sentences, and overlapping windows mean the same
    # occurrence is seen more than once, so the result must be de-duplicated and expressed
    # in DOCUMENT coordinates.
    text = " ".join(f"Sentence number {i} mentions metformin." for i in range(10))

    def fake_run(chunk: str) -> list[dict]:
        spans = []
        cursor = chunk.find("metformin")
        while cursor != -1:
            spans.append(
                {
                    "entity_group": "Chemical",
                    "score": 0.9,
                    "word": "metformin",
                    "start": cursor,
                    "end": cursor + len("metformin"),
                }
            )
            cursor = chunk.find("metformin", cursor + 1)
        return spans

    spans = predict_windowed(text, run=fake_run, count_tokens=_chars, max_tokens=100)

    expected_starts = []
    cursor = text.find("metformin")
    while cursor != -1:
        expected_starts.append(cursor)
        cursor = text.find("metformin", cursor + 1)

    assert [s["start"] for s in spans] == expected_starts
    assert all(text[s["start"] : s["end"]] == "metformin" for s in spans)
