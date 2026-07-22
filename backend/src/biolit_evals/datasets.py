import json
from typing import Any, cast

from biolit.domain.records import Entity
from biolit.ner.labels import CHEMICAL, DISEASE, canonical_label


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
                if label not in (CHEMICAL, DISEASE):
                    raise ValueError(f"non-canonical label in {rec.get('pmid')}: {label}")
                surface = text[start:end]
                recorded_text = ent.get("text")
                if recorded_text is not None and recorded_text != surface:
                    record_id = rec.get("pmid") or rec.get("paper_id")
                    raise ValueError(
                        f"gold span text mismatch in {record_id}: recorded text "
                        f"{recorded_text!r} does not match text[{start}:{end}] = {surface!r}"
                    )
                entities.append(Entity(text=surface, label=label, start=start, end=end))
            pairs.append((text, entities))
    return pairs


def load_bc5cdr_test() -> list[tuple[str, list[Entity]]]:
    """Load the BC5CDR test split from `tner/bc5cdr` (lazy heavy import)."""
    from datasets import load_dataset

    ds = load_dataset("tner/bc5cdr", split="test")
    # This is the `tner/bc5cdr` *dataset's* label encoding (its tag-id -> BIO-label map),
    # not the NER model's. The model's own config.json output head uses a different id
    # map ({0:"O",1:"B-Chemical",2:"I-Chemical",3:"B-Disease",4:"I-Disease"}) — these are
    # two unrelated encodings and are NOT supposed to match. Do not "fix" this map to
    # align with the model's config; that would silently corrupt every gold span. This
    # map is verified against the real `ds.features` before the benchmark is run.
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
