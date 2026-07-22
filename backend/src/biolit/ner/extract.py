from biolit.domain.records import Entity
from biolit.ner.labels import canonical_label
from biolit.ner.model import NerModel


def extract_entities(text: str, model: NerModel, *, score_threshold: float = 0.5) -> list[Entity]:
    """Run the NER model over `text` and return canonical CHEMICAL/DISEASE entities.

    Drops spans whose label is not canonical or whose score is below threshold.
    Returns entities sorted by start offset. Empty/whitespace text -> [].
    """
    if not text or not text.strip():
        return []
    entities: list[Entity] = []
    for span in model.predictor(text):
        raw = span.get("entity_group") or span.get("entity") or ""
        label = canonical_label(raw)
        if label is None:
            continue
        if float(span.get("score", 1.0)) < score_threshold:
            continue
        start = span.get("start")
        end = span.get("end")
        word = span.get("word")
        if word is None and start is not None and end is not None:
            word = text[start:end]
        entities.append(Entity(text=word or "", label=label, start=start, end=end))
    entities.sort(key=lambda e: e.start if e.start is not None else 0)
    return entities
