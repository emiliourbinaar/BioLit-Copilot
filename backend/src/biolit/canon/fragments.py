import re
from dataclasses import dataclass

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity

# Hyphen/dash/slash connectors that indicate a split token rather than a phrase boundary.
# Whitespace is deliberately excluded: "GLP 1RA" is two words, "GLP-1RA" is one split token.
_CONNECTORS = frozenset("-‐‑‒–/")
_PLAIN_WORD = re.compile(r"[A-Za-z]{3,}$")


@dataclass(frozen=True)
class FragmentCandidate:
    text: str
    label: EntityLabel
    start: int
    end: int
    source_indices: tuple[int, ...]


def _connector_only(sep: str) -> bool:
    return all(ch in _CONNECTORS for ch in sep)  # all("") is True -> zero-gap counts


def _is_fragment_shaped(token: str) -> bool:
    """A plain lowercase/Titlecase word (>=3 letters) is a standalone term; everything
    else (has a digit, is all-caps, or is <3 chars) looks like a tokenizer fragment."""
    if _PLAIN_WORD.match(token) and not token.isupper():
        return False
    return True


def merge_fragments(entities: list[Entity], text: str) -> list[FragmentCandidate]:
    order = sorted(range(len(entities)), key=lambda i: entities[i].start or 0)
    candidates: list[FragmentCandidate] = []
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n:
            a = entities[order[j]]
            b = entities[order[j + 1]]
            if a.label != b.label or a.end is None or b.start is None:
                break
            if not _connector_only(text[a.end : b.start]):
                break
            j += 1
        run = order[i : j + 1]
        if len(run) >= 2 and any(_is_fragment_shaped(entities[k].text) for k in run):
            start = entities[run[0]].start
            end = entities[run[-1]].end
            if start is not None and end is not None:
                candidates.append(
                    FragmentCandidate(
                        text=text[start:end],
                        label=entities[run[0]].label,
                        start=start,
                        end=end,
                        source_indices=tuple(run),
                    )
                )
        i = j + 1
    return candidates
