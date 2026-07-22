import json
from typing import Any, cast

from biolit.domain.records import Entity
from biolit.ner.labels import canonical_label


def bio_tags_to_spans(tokens: list[str], tags: list[str]) -> tuple[str, list[Entity]]:
    """Join tokens with single spaces and convert BIO tags to char-offset entities."""
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(tok)
        cursor += len(tok)
        offsets.append((start, cursor))
    text = "".join(text_parts)

    entities: list[Entity] = []
    cur_label: str | None = None
    cur_start = 0
    cur_end = 0

    def flush() -> None:
        nonlocal cur_label
        if cur_label is not None:
            entities.append(
                Entity(text=text[cur_start:cur_end], label=cur_label, start=cur_start, end=cur_end)
            )
            cur_label = None

    for (start, end), tag in zip(offsets, tags, strict=True):
        label = canonical_label(tag)
        if label is None:  # "O" or an out-of-scope type closes any open span
            flush()
            continue
        # A B- tag always opens a new entity (so adjacent same-label entities stay
        # separate); a type change mid-span closes the previous one too.
        if cur_label is not None and (tag.upper().startswith("B-") or label != cur_label):
            flush()
        if cur_label is None:
            cur_label, cur_start, cur_end = label, start, end
        else:
            cur_end = end
    flush()
    return text, entities


def load_domain_sample(path: str) -> list[tuple[str, list[Entity]]]:
    """Parse the blind-annotated domain gold JSONL (with provenance) into (text, entities)."""
    pairs: list[tuple[str, list[Entity]]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]
            entities: list[Entity] = []
            for ent in rec.get("entities", []):
                start, end, label = ent["start"], ent["end"], ent["label"]
                if not (0 <= start < end <= len(text)):
                    raise ValueError(f"span out of bounds in {rec.get('pmid')}: {ent}")
                if label not in ("CHEMICAL", "DISEASE"):
                    raise ValueError(f"non-canonical label in {rec.get('pmid')}: {label}")
                entities.append(Entity(text=text[start:end], label=label, start=start, end=end))
            pairs.append((text, entities))
    return pairs


def load_bc5cdr_test() -> list[tuple[str, list[Entity]]]:
    """Load the BC5CDR test split from `tner/bc5cdr` (lazy heavy import)."""
    from datasets import load_dataset

    ds = load_dataset("tner/bc5cdr", split="test")
    id2label = {
        0: "O",
        1: "B-Chemical",
        2: "B-Disease",
        3: "I-Disease",
        4: "I-Chemical",
    }
    pairs: list[tuple[str, list[Entity]]] = []
    for raw_row in ds:
        row = cast(dict[str, Any], raw_row)
        tokens = row["tokens"]
        tags = [id2label[t] if isinstance(t, int) else t for t in row["tags"]]
        pairs.append(bio_tags_to_spans(tokens, tags))
    return pairs
