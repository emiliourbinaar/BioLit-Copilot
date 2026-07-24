import re
from collections.abc import Callable

# Sentence boundary: terminal punctuation followed by whitespace. Deliberately simple --
# it is a *packing hint*, not a claim about linguistics. Mis-splits (e.g. "0.5 mg",
# "e.g. metformin") are made harmless by the one-sentence overlap between windows.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into (start, end) spans at sentence boundaries, covering it exactly."""
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


def _split_oversized(
    text: str, span: tuple[int, int], count_tokens: Callable[[str], int], max_tokens: int
) -> list[tuple[int, int]]:
    """Break a single span that exceeds the budget on its own, at whitespace.

    Only reachable for a sentence longer than the whole budget, which is pathological in
    abstracts but must not crash: the caller's overlap cannot help here, so this splits
    greedily at word boundaries and accepts that a word-level split point could in
    principle land inside an entity.
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
    return pieces


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

    `count_tokens` is injected so this stays a pure function, testable with no tokenizer.
    """
    if not text:
        return []
    sentences = _sentence_spans(text)

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
    that window, so every offset is shifted back by the window's start. Windows overlap, so
    the same span is usually predicted more than once -- results are de-duplicated on
    (start, end, group), keeping the highest-scoring copy, so an entity seen whole in one
    window wins over a partial sighting at another window's edge.

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
    return sorted(best.values(), key=lambda s: (s.get("start") or 0, s.get("end") or 0))
