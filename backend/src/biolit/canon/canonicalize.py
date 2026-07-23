from biolit.canon.fragments import merge_fragments
from biolit.canon.linker import Linker
from biolit.domain.records import Entity


def canonicalize(entities: list[Entity], text: str, *, linker: Linker) -> list[Entity]:
    """Populate canonical_id/canonical_name on each entity via MeSH linking.

    Merged fragment candidates are tried first: when one links, every entity it spans
    inherits that concept (the ADR-0008 fix). Entities not resolved via a merge fall back
    to an individual lookup on their own surface form. Unlinked entities stay NIL (None).
    """
    resolved: dict[int, tuple[str, str]] = {}
    for cand in merge_fragments(entities, text):
        result = linker.link(cand.text)
        if result.concept is not None:
            for idx in cand.source_indices:
                resolved[idx] = (result.concept.id, result.concept.name)

    out: list[Entity] = []
    for i, entity in enumerate(entities):
        if i in resolved:
            cid, cname = resolved[i]
        else:
            r = linker.link(entity.text)
            cid, cname = (r.concept.id, r.concept.name) if r.concept is not None else (None, None)
        out.append(entity.model_copy(update={"canonical_id": cid, "canonical_name": cname}))
    return out
