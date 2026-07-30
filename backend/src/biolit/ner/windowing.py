import re
from collections.abc import Callable

# Sentence boundary: terminal punctuation followed by whitespace. Deliberately simple --
# it is a *packing hint*, not a claim about linguistics. Mis-splits (e.g. "0.5 mg",
# "e.g. metformin") are made harmless by the one-sentence overlap between windows.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into (start, end) spans at sentence boundaries.

    The spans do NOT cover `text` exactly: the whitespace separating two sentences belongs
    to neither span, so `sentence_spans("A b. C d.") == [(0, 4), (5, 9)]` leaves position 4
    uncovered. That is load-bearing, not incidental -- `SameSentencePairing` and
    `pairing_diagnostics` both treat an entity starting in a gap as unplaceable, and each
    has a test pinning it. `test_sentence_spans_leave_the_inter_sentence_separator_uncovered`
    owns this contract.

    Public because it has a second consumer: `biolit.cluster.pairing.SameSentencePairing`.
    The splitter is deliberately simple and mis-splits abbreviations ("e.g. metformin").
    In windowing that is harmless (one-sentence overlap absorbs it); in pairing a mis-split
    can drop a real pair or invent one, which is a measured cost of the heuristic, not a bug
    to hide -- both strategies are reported side by side precisely so it is visible.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start()
        if end > cursor:
            spans.append((cursor, end))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans or [(0, len(text))]


def _hard_split(
    text: str, start: int, end: int, count_tokens: Callable[[str], int], max_tokens: int
) -> list[tuple[int, int]]:
    """Split [start, end) at character granularity so every piece fits the budget.

    Last-resort path for a run with no whitespace to split on (a long identifier or a
    malformed document). Splitting mid-word can cut an entity, but the alternative is
    handing the model an over-budget window, which raises a tensor-size error and loses the
    whole document.
    """
    pieces: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        lo, hi, fit = cursor + 1, end, cursor + 1
        while lo <= hi:  # bisect for the longest prefix that still fits
            mid = (lo + hi) // 2
            if count_tokens(text[cursor:mid]) <= max_tokens:
                fit, lo = mid, mid + 1
            else:
                hi = mid - 1
        pieces.append((cursor, fit))
        cursor = fit
    return pieces


def _split_oversized(
    text: str, span: tuple[int, int], count_tokens: Callable[[str], int], max_tokens: int
) -> list[tuple[int, int]]:
    """Break a single span that exceeds the budget on its own, at whitespace.

    Only reachable for a sentence longer than the whole budget, which is pathological in
    abstracts but must not crash: the caller's overlap cannot help here, so this splits
    greedily at word boundaries and accepts that a word-level split point could in
    principle land inside an entity. Any resulting piece that STILL exceeds the budget
    (a whitespace-free run longer than the budget) is split at character granularity, so
    this function's postcondition -- every piece fits -- always holds.
    """
    start, end = span
    pieces: list[tuple[int, int]] = []
    word_starts = [start] + [start + m.end() for m in re.finditer(r"\s+", text[start:end])]
    piece_start = start
    last_fit = start
    for ws in word_starts[1:] + [end]:
        if count_tokens(text[piece_start:ws]) > max_tokens and last_fit > piece_start:
            pieces.append((piece_start, last_fit))
            piece_start = last_fit
        last_fit = ws
    if piece_start < end:
        pieces.append((piece_start, end))
    resolved: list[tuple[int, int]] = []
    for piece in pieces:
        if count_tokens(text[piece[0] : piece[1]]) > max_tokens:
            resolved.extend(_hard_split(text, piece[0], piece[1], count_tokens, max_tokens))
        else:
            resolved.append(piece)
    return resolved


def plan_windows(
    text: str,
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_sentences: int = 1,
) -> list[tuple[int, int]]:
    """Plan (start, end) char windows over `text` that each fit within `max_tokens`.

    Windows are packed from whole sentences, because an entity practically never spans a
    sentence boundary -- so a boundary is the safest place to cut. Consecutive windows
    additionally overlap by `overlap_sentences` sentences, which covers the case where the
    simple sentence regex splits in the wrong place (an abbreviation or a decimal) and
    would otherwise have cut an entity in half. Overlap makes spans appear twice; the
    caller is responsible for de-duplicating them.

    **The overlap is conditional, not universal:** it only materializes when a window holds
    more than one sentence, since a window holding a single sentence cannot step back
    without failing to advance. At the production budget (450 tokens against sentences of
    a few dozen tokens) windows hold many sentences and every internal boundary is spanned;
    a single sentence filling an entire window is pathological, and in that case its
    boundaries are hard cuts.

    `count_tokens` is injected so this stays a pure function, testable with no tokenizer.
    """
    if not text:
        return []
    sentences = sentence_spans(text)

    # Expand any single sentence that cannot fit on its own.
    units: list[tuple[int, int]] = []
    for span in sentences:
        if count_tokens(text[span[0] : span[1]]) > max_tokens:
            units.extend(_split_oversized(text, span, count_tokens, max_tokens))
        else:
            units.append(span)

    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        start = units[i][0]
        j = i
        while j + 1 < len(units) and count_tokens(text[start : units[j + 1][1]]) <= max_tokens:
            j += 1
        windows.append((start, units[j][1]))
        if j + 1 >= len(units):
            break
        # Step back `overlap_sentences` units so a mis-placed boundary is still covered
        # whole by the next window -- but always advance, or this loops forever.
        i = max(j + 1 - overlap_sentences, i + 1)
    return windows


def predict_windowed(
    text: str,
    *,
    run: Callable[[str], list[dict]],
    count_tokens: Callable[[str], int],
    max_tokens: int,
) -> list[dict]:
    """Run `run` over `text` in windows and return spans in DOCUMENT coordinates.

    `run` is the model call; it sees one window at a time and reports offsets relative to
    that window, so every offset is shifted back by the window's start.

    Overlapping windows see the same text twice, which produces two kinds of repeat:
    identical spans (same start, end and group -- de-duplicated, keeping the
    highest-scoring copy), and spans where one window caught an entity whole while another
    caught it clipped at a window edge. The clipped one is strictly CONTAINED in the whole
    one, and both would otherwise survive de-duplication because their end offsets differ,
    emitting nested spans that cannot occur on single-window input. So a span strictly
    contained in another span of the same group is dropped. Two separate occurrences of the
    same surface form are not contained in one another and both survive.

    Both `run` and `count_tokens` are injected, keeping this pure and testable with no
    model or tokenizer.
    """
    if not text or not text.strip():
        return []
    windows = plan_windows(text, count_tokens=count_tokens, max_tokens=max_tokens)
    if len(windows) == 1 and windows[0] == (0, len(text)):
        return list(run(text))  # fast path: nearly all inputs fit in one window

    best: dict[tuple[int | None, int | None, str], dict] = {}
    for start, _end in windows:
        for span in run(text[start:_end]):
            shifted = dict(span)
            if shifted.get("start") is not None:
                shifted["start"] = shifted["start"] + start
            if shifted.get("end") is not None:
                shifted["end"] = shifted["end"] + start
            key = (
                shifted.get("start"),
                shifted.get("end"),
                str(shifted.get("entity_group") or shifted.get("entity") or ""),
            )
            previous = best.get(key)
            if previous is None or float(shifted.get("score", 0.0)) > float(
                previous.get("score", 0.0)
            ):
                best[key] = shifted

    spans = sorted(best.values(), key=lambda s: (s.get("start") or 0, s.get("end") or 0))
    return [s for s in spans if not _is_contained(s, spans)]


def _is_contained(span: dict, spans: list[dict]) -> bool:
    """True if `span` lies strictly inside another span of the same group."""
    start, end = span.get("start"), span.get("end")
    if start is None or end is None:
        return False
    group = span.get("entity_group") or span.get("entity")
    for other in spans:
        if other is span:
            continue
        if (other.get("entity_group") or other.get("entity")) != group:
            continue
        o_start, o_end = other.get("start"), other.get("end")
        if o_start is None or o_end is None:
            continue
        if o_start <= start and end <= o_end and (o_start, o_end) != (start, end):
            return True
    return False
