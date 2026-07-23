import json
from dataclasses import dataclass

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


def load_domain_norm_sample(path: str) -> list[GoldMention]:
    mentions: list[GoldMention] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pmid = str(rec.get("pmid") or rec.get("paper_id") or "")
            for ent in rec.get("entities", []):
                label = canonical_label(ent["label"])
                if label is None:
                    raise ValueError(f"non-canonical label in {pmid}: {ent['label']}")
                mentions.append(
                    GoldMention(
                        pmid=pmid,
                        start=int(ent["start"]),
                        end=int(ent["end"]),
                        text=ent["text"],
                        label=label,
                        mesh_ids=reconcile_mesh_id(str(ent["mesh_id"])),
                    )
                )
    return mentions
