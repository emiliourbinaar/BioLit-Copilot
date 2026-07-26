"""Precondition guard for document-scoped canonicalization mechanisms.

Some mechanisms need the WHOLE source document, not just the span neighbourhood: in-document
abbreviation expansion has to find the defining `long form (SHORT)` occurrence, which is
typically sentences away from the mentions it licenses. Handed a fragment, such a mechanism
finds no definitions and silently returns nothing -- indistinguishable from "this document
had no abbreviations".

This project has now hit undetected precondition violations three times (the 512-token
truncation crash, the unconditional merge, the CTD column schema), so the precondition is
checked and reported rather than only documented.
"""

import re
from dataclasses import dataclass

from biolit.domain.records import Entity

# The MULTI-SENTENCE check is the discriminator; the length floor is only a backstop
# against degenerate input. Measured on the real corpora: requiring one sentence break
# separates them perfectly on its own -- 0 of 500 BC5CDR abstracts flagged, 49 of 49 domain
# sentences flagged. The length floor adds nothing at any value up to 200 and starts
# producing FALSE POSITIVES at 250 (2 of 500 real abstracts), so it is set well below the
# shortest real abstract observed (204 chars) rather than near it.
_MIN_DOCUMENT_CHARS = 100
_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]?\s+\S")


@dataclass(frozen=True)
class ContextCheck:
    """Whether `text` can support a document-scoped mechanism.

    `ok=False` splits into two kinds of failure, both reported the same way but very
    different in character:

    - **Provable** -- spans out of bounds, or not slicing to their own text. The caller
      passed text that is definitively not the document these entities came from.
    - **Heuristic** -- the text is shaped like a fragment. A sentence and the entities
      extracted from it are perfectly self-consistent, so no provable check can catch this
      one; only the shape of the text can.
    """

    ok: bool
    reason: str | None = None


def check_document_context(entities: list[Entity], text: str) -> ContextCheck:
    for e in entities:
        if e.start is None or e.end is None:
            continue
        if e.start < 0 or e.end > len(text):
            return ContextCheck(
                False, f"entity span {e.start}:{e.end} lies outside text of length {len(text)}"
            )
        if text[e.start : e.end] != e.text:
            return ContextCheck(
                False,
                f"entity span {e.start}:{e.end} does not slice to its text "
                f"({text[e.start : e.end]!r} != {e.text!r})",
            )
    if len(text) < _MIN_DOCUMENT_CHARS or len(_SENTENCE_BREAK.findall(text)) < 1:
        return ContextCheck(
            False,
            f"insufficient context: {len(text)} chars, "
            f"{len(_SENTENCE_BREAK.findall(text)) + 1} sentence(s) -- looks like a fragment, "
            "not a whole document",
        )
    return ContextCheck(True)
