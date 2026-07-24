import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from biolit.domain.enums import EntityLabel
from biolit.ner.labels import canonical_label


@dataclass(frozen=True)
class GoldMention:
    pmid: str
    start: int
    end: int
    text: str
    label: EntityLabel
    mesh_ids: tuple[str, ...]  # prefixed ("MESH:D011085"); () == unlinkable


def reconcile_mesh_id(raw: str) -> tuple[str, ...]:
    """Reconcile a BC5CDR/domain gold id string to prefixed MeSH/OMIM ids.

    BC5CDR ids are bare accessions (`D011085`) with `-1` for unlinkable and `|` joining
    composite mentions; CTD/domain ids may already carry a `MESH:`/`OMIM:` prefix.
    """
    ids: list[str] = []
    for part in raw.split("|"):
        part = part.strip()
        if not part or part == "-1":
            continue
        ids.append(part if ":" in part else f"MESH:{part}")
    return tuple(ids)


def parse_pubtator(text: str) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6:
            continue  # title/abstract (`PMID|t|…`) and blank lines have no tabs
        pmid, start, end, mention, raw_type, raw_id = parts[:6]
        label = canonical_label(raw_type)
        if label is None:
            continue  # skip out-of-scope types (e.g. Gene)
        mentions.append(
            GoldMention(
                pmid=pmid,
                start=int(start),
                end=int(end),
                text=mention,
                label=label,
                mesh_ids=reconcile_mesh_id(raw_id),
            )
        )
    return mentions


@dataclass(frozen=True)
class GoldDocument:
    pmid: str
    text: str
    mentions: list[GoldMention]


def parse_pubtator_documents(text: str) -> list[GoldDocument]:
    """Parse a PubTator dump into documents carrying their own text and mentions.

    Document text is `title + " " + abstract`: PubTator offsets are expressed against that
    concatenation. Verified against the real corpus -- 9809 of 9809 mentions across all 500
    BC5CDR test documents satisfy `text[start:end] == mention text` under it, and a
    zero-length separator breaks alignment. Mention parsing delegates to `parse_pubtator`,
    which already ignores the title/abstract lines, so the two parsers cannot disagree
    about a mention -- only about document segmentation.
    """
    documents: list[GoldDocument] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        pmid = ""
        title = ""
        abstract = ""
        for line in block.splitlines():
            if not title and "|t|" in line:
                pmid, title = line.split("|t|", 1)
            elif not abstract and "|a|" in line:
                _, abstract = line.split("|a|", 1)
        if not pmid:
            continue
        documents.append(
            GoldDocument(pmid=pmid, text=title + " " + abstract, mentions=parse_pubtator(block))
        )
    return documents


def _mentions_from_record(rec: dict[str, Any], pmid: str) -> list[GoldMention]:
    """Parse and validate one domain-gold JSONL record's entities.

    Shared by both domain loaders so the bounds and span/text checks -- which are what stop
    an annotator off-by-one from silently corrupting the gold -- cannot drift apart.
    """
    text = rec["text"]
    mentions: list[GoldMention] = []
    for ent in rec.get("entities", []):
        start, end = int(ent["start"]), int(ent["end"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"span out of bounds in {pmid}: {ent}")
        label = canonical_label(ent["label"])
        if label is None:
            raise ValueError(f"non-canonical label in {pmid}: {ent['label']}")
        surface = text[start:end]
        recorded_text = ent["text"]
        if recorded_text != surface:
            raise ValueError(
                f"gold span text mismatch in {pmid}: recorded text "
                f"{recorded_text!r} does not match text[{start}:{end}] = {surface!r}"
            )
        mentions.append(
            GoldMention(
                pmid=pmid,
                start=start,
                end=end,
                text=surface,
                label=label,
                mesh_ids=reconcile_mesh_id(str(ent["mesh_id"])),
            )
        )
    return mentions


def _iter_domain_records(path: str) -> Iterator[tuple[str, dict[str, Any]]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield str(rec.get("pmid") or rec.get("paper_id") or ""), rec


def load_domain_norm_sample(path: str) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    for pmid, rec in _iter_domain_records(path):
        mentions.extend(_mentions_from_record(rec, pmid))
    return mentions


def load_domain_norm_documents(path: str) -> list[GoldDocument]:
    """Load the blind domain normalization gold as documents (one per annotated sentence)."""
    return [
        GoldDocument(pmid=pmid, text=rec["text"], mentions=_mentions_from_record(rec, pmid))
        for pmid, rec in _iter_domain_records(path)
    ]
