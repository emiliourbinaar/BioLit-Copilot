"""Build the {paper_id: [concept_id, ...]} file the `overlap` baseline requires.

Runs the real Phase 2/3 path -- `extract_entities` -> `canonicalize` -> `canonical_id` --
over the corpus abstracts, which is the same path `end_to_end` scores. The concepts are
therefore PREDICTED, not read off the gold manifest, and that distinction is the whole reason
this file exists: the scoring CLI refuses a manifest-derived fallback because a pair's two
papers share exactly one curated key by construction, so a manifest-built concept set is
identical between them and ConceptOverlapCritic would answer `agreement` on every pair.

`main()` only; per this project's convention it gets no direct unit test, and every function
it calls is tested in its own module.
"""

DEFAULT_TEXTS = "data/contradiction_abstract_texts.json"
DEFAULT_OUT = "data/contradiction_concepts.json"


def main(argv: list[str] | None = None) -> None:
    # Heavy imports local to main, same pattern as end_to_end.main.
    import argparse
    import json
    from pathlib import Path

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel

    parser = argparse.ArgumentParser(description="Predict canonical concepts per paper.")
    parser.add_argument("--texts", default=DEFAULT_TEXTS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    texts: dict[str, str] = json.loads(Path(args.texts).read_text(encoding="utf-8"))
    settings = get_settings()
    dictionary = MeshDictionary.from_artifact(settings.mesh_artifact_path)
    linker = DictionaryLinker(dictionary)
    model = NerModel.load(settings)

    concepts: dict[str, list[str]] = {}
    for i, (pmid, text) in enumerate(sorted(texts.items()), start=1):
        entities = extract_entities(text, model, score_threshold=settings.ner_score_threshold)
        canon = canonicalize(entities, text, linker=linker)
        # Sorted for determinism: the file is an input to a scored run, so it must not depend
        # on set iteration order.
        concepts[pmid] = sorted({e.canonical_id for e in canon if e.canonical_id})
        if i % 250 == 0:
            print(f"  {i}/{len(texts)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(concepts), encoding="utf-8")

    linked = sum(1 for ids in concepts.values() if ids)
    total = sum(len(ids) for ids in concepts.values())
    print(f"wrote {out}: {len(concepts)} papers, {linked} with >=1 concept, {total} concepts")
    print(f"mean concepts/paper: {total / len(concepts):.2f}")


if __name__ == "__main__":
    main()
