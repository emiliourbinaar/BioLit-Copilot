from biolit.canon.fragments import merge_fragments
from biolit.canon.linker import Linker
from biolit.domain.records import Entity


def canonicalize(entities: list[Entity], text: str, *, linker: Linker) -> list[Entity]:
    """Populate canonical_id/canonical_name on each entity via MeSH linking.

    Each entity is looked up on its own surface first. A merged fragment candidate is
    then applied ONLY to constituents that did not link individually (the ADR-0008 case:
    `GLP` and `1RA` are each meaningless alone). This ordering means a spurious merge --
    the heuristic in `merge_fragments` is deliberately permissive -- can never overwrite a
    correct individual link with one wrong shared concept. Unlinked entities stay NIL.
    """
    individual = [linker.link(entity.text) for entity in entities]
    resolved: dict[int, tuple[str, str]] = {}
    for cand in merge_fragments(entities, text):
        result = linker.link(cand.text)
        if result.concept is None:
            continue
        for idx in cand.source_indices:
            if individual[idx].concept is None:
                resolved[idx] = (result.concept.id, result.concept.name)

    out: list[Entity] = []
    for i, entity in enumerate(entities):
        if i in resolved:
            cid, cname = resolved[i]
        else:
            concept = individual[i].concept
            cid, cname = (concept.id, concept.name) if concept is not None else (None, None)
        out.append(entity.model_copy(update={"canonical_id": cid, "canonical_name": cname}))
    return out
