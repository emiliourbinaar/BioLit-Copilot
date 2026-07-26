"""In-document abbreviation detection: `long form (SHORT)`.

Measurement-only. This exists to size how much of the exact-span linking gap is reachable
by a deterministic mechanism -- these abstracts overwhelmingly define their abbreviations on
first use -- before any embedding fallback is scoped. It is a simplified Schwartz-Hearst:
the short form's letters must appear in order in the candidate long form, and the long form
must begin with the short form's first letter at a word boundary.
"""

import re

_PARENTHETICAL = re.compile(r"\(([^()]{2,15})\)")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")


def _is_short_form(s: str) -> bool:
    """Abbreviation-shaped: 2-10 chars, no whitespace, contains a letter, starts alphanumeric."""
    return (
        2 <= len(s) <= 10
        and not any(c.isspace() for c in s)
        and any(c.isalpha() for c in s)
        and s[0].isalnum()
    )


def _letters_in_order(long_form: str, short: str) -> bool:
    """Every alphanumeric char of `short` appears in `long_form`, in order."""
    i = 0
    lowered = long_form.lower()
    for c in short.lower():
        if not c.isalnum():
            continue
        i = lowered.find(c, i)
        if i < 0:
            return False
        i += 1
    return True


def _best_long_form(preceding: str, short: str) -> str | None:
    """Shortest trailing word-run of `preceding` that can define `short`.

    Schwartz-Hearst bounds the search to roughly len(short)+5 words; the run must start with
    the short form's first letter, which is what stops an arbitrary span of preceding text
    from being accepted as a definition.
    """
    words = [(m.start(), m.group(0)) for m in _WORD.finditer(preceding)]
    if not words:
        return None
    first = next((c for c in short.lower() if c.isalnum()), "")
    max_words = min(len(short) + 5, len(words))
    for n in range(1, max_words + 1):
        start = words[-n][0]
        candidate = preceding[start:].strip()
        if not candidate.lower().startswith(first):
            continue
        if _letters_in_order(candidate, short):
            return candidate
    return None


def find_abbreviations(text: str) -> dict[str, str]:
    """Map each defined short form to its long form. First definition wins."""
    out: dict[str, str] = {}
    for m in _PARENTHETICAL.finditer(text):
        short = m.group(1).strip()
        if not _is_short_form(short):
            continue
        long_form = _best_long_form(text[: m.start()].rstrip(), short)
        if long_form and short not in out:
            out[short] = long_form
    return out
