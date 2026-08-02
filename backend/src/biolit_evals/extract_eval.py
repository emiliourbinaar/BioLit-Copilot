from collections.abc import Mapping, Sequence

from biolit.cluster.pairing import sentence_index
from biolit.domain.enums import EntityLabel
from biolit.ner.windowing import sentence_spans
from biolit_evals.mesh_gold import GoldDocument


def gold_finding_sentences(
    documents: Sequence[GoldDocument],
    relations: Mapping[str, set[tuple[str, str]]],
) -> dict[str, set[int]]:
    """Sentences where BOTH endpoints of at least one gold CID relation are gold-annotated.

    PROXY, NOT GROUND TRUTH. BC5CDR annotates CID relations at DOCUMENT level; sentence-level
    co-occurrence is this project's inference about where the relation is asserted. Some
    qualifying sentences state background rather than a finding, and a paper's actual key
    finding may concern efficacy, which CID does not annotate at all. A high score against
    this gold means "selects sentences containing the annotated relation" -- NOT "selects the
    paper's key finding". Cite it that way.

    The resulting sentence COUNT is a regression pin, not an independent validation: no
    published corpus statistic exists for it. The independent invariant is that the number
    of relations with >=1 gold sentence is <= 1066 (BC5CDR Test-500's published gold CID
    relation count) -- every gold sentence traces back to a real relation, never invented.
    """
    gold: dict[str, set[int]] = {}
    for document in documents:
        pairs = relations.get(document.pmid)
        if not pairs:
            continue
        spans = sentence_spans(document.text)
        by_label: dict[EntityLabel, dict[int, set[str]]] = {
            EntityLabel.CHEMICAL: {},
            EntityLabel.DISEASE: {},
        }
        for mention in document.mentions:
            index = sentence_index(spans, mention.start)
            if index is None or mention.label not in by_label:
                continue
            by_label[mention.label].setdefault(index, set()).update(mention.mesh_ids)
        chemicals = by_label[EntityLabel.CHEMICAL]
        diseases = by_label[EntityLabel.DISEASE]
        for index in set(chemicals) & set(diseases):
            if any(c in chemicals[index] and d in diseases[index] for c, d in pairs):
                gold.setdefault(document.pmid, set()).add(index)
    return gold
