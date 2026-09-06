"""Export the DEF-0001 acronym pairs as a blind adjudication sheet.

Design: `docs/superpowers/specs/2026-09-06-acronym-adjudication-design.md`.

The population is a CENSUS of the eight frozen states -- every distinct (surface, concept)
pair whose surface is alphabetic, all-caps and at most four characters -- not a sample, so
nothing here computes an interval and nothing should.
"""

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

#: §2. `granularity` is a separate label rather than a flavour of `wrong` because DEF-0002's
#: failure shape -- a bare parent concept standing in for its specific child, `ICH` linked to
#: `Cerebral Hemorrhage` where the text says intracranial haemorrhage -- is a different defect
#: with a different fix, and collapsing it would file it as a plain mislink.
ACRONYM_LABELS = frozenset({"correct", "wrong", "granularity", "cant_tell"})


@dataclass(frozen=True)
class PairEvidence:
    """One (surface, concept) pair with everything needed to judge it, gathered from state."""

    surface: str
    concept_id: str
    concept_name: str
    aliases: tuple[str, ...]
    contexts: tuple[str, ...]
    n_mentions: int
    type_violation: bool


@dataclass(frozen=True)
class AcronymRow:
    row_id: str
    surface: str
    concept_id: str
    concept_name: str
    aliases: tuple[str, ...]
    contexts: tuple[str, ...]
    n_mentions: int
    type_violation: bool
    is_control: bool


#: §1. Alphabetic, all-caps, at most four characters -- the class DEF-0001 defines. Kept as one
#: predicate so the export and any later re-measurement cannot drift apart on the population.
def is_target_surface(text: str) -> bool:
    return len(text) <= 4 and text.isupper() and text.isalpha()


#: Characters either side of the mention. Wide enough to carry a first-use gloss, which sits
#: immediately BEFORE the parenthesised acronym and can run long ("2,3,4-tri-O-acetyl...").
_WINDOW = 170


def _window(abstract: str, start: int, end: int) -> str:
    """A verbatim slice of the abstract, trimmed to whitespace so it does not cut a word.

    ⛔ Sliced from the stored abstract by offset, never rebuilt from the entity's own `text`.
    Reconstructing it would make any later "the context contains the surface" check
    tautological -- the shape that made an earlier gold loader's validation vacuous end to end.
    """
    left = abstract.rfind(" ", 0, max(0, start - _WINDOW))
    right = abstract.find(" ", min(len(abstract), end + _WINDOW))
    return abstract[(left + 1) if left != -1 else 0 : right if right != -1 else len(abstract)]


def gather_pairs(
    states: Iterable[Mapping[str, Any]],
    *,
    concept_labels: Mapping[str, Sequence[str]],
    aliases: Mapping[str, Sequence[str]] | None = None,
    max_contexts: int = 2,
) -> list[PairEvidence]:
    """Reduce the frozen states to one row per distinct (surface, concept) pair.

    A pair is flagged `type_violation` when ANY of its mentions carries an NER label the
    concept's own type set excludes. Any rather than all, deliberately: `CD` and `DIC` are
    labelled differently in different documents, and a pair that is inconsistent in even one
    place is one the linker had the evidence to refuse.
    """
    contexts: dict[tuple[str, str], dict[str, str]] = {}
    names: dict[tuple[str, str], str] = {}
    counts: Counter[tuple[str, str]] = Counter()
    violations: set[tuple[str, str]] = set()

    for state in states:
        papers = {paper["id"]: paper for paper in state["candidate_papers"]}
        for paper_id, record in state["extracted_records"].items():
            abstract = papers[paper_id].get("abstract") or ""
            for entity in record["entities"]:
                concept_id = entity["canonical_id"]
                if not concept_id or not is_target_surface(entity["text"]):
                    continue
                key = (entity["text"], concept_id)
                counts[key] += 1
                names.setdefault(key, entity["canonical_name"] or concept_id)
                types = concept_labels.get(concept_id)
                if types is not None and entity["label"] not in types:
                    violations.add(key)
                # ⚠️ ONE WINDOW PER PAPER. Two mentions in one abstract are usually a few
                # words apart, so their windows overlap and the annotator is shown the same
                # sentence twice -- half the evidence budget spent on nothing. It also skews
                # `cant_tell`: a pair whose meaning is settled in a SECOND paper would read
                # as unjudgeable purely because both windows came from the first.
                bucket = contexts.setdefault(key, {})
                if paper_id not in bucket and len(bucket) < max_contexts:
                    bucket[paper_id] = _window(abstract, entity["start"], entity["end"])

    return [
        PairEvidence(
            surface=surface,
            concept_id=concept_id,
            concept_name=names[(surface, concept_id)],
            aliases=tuple((aliases or {}).get(concept_id, ())),
            contexts=tuple(contexts[(surface, concept_id)].values()),
            n_mentions=counts[(surface, concept_id)],
            type_violation=(surface, concept_id) in violations,
        )
        for surface, concept_id in sorted(counts)
    ]


def choose_controls(pairs: list[PairEvidence], *, n: int, seed: int) -> list[PairEvidence]:
    """Deliberately mispaired rows: a real surface with a real concept from a different pair.

    The surface keeps its OWN contexts. A control wearing the donor's contexts would be
    rejectable from the surface/text mismatch alone, without the annotator ever reading the
    concept -- which is not the discrimination Gate 2 claims to measure.

    ⚠️ The donor concept is drawn from a pair with a DIFFERENT surface, and the result is
    checked against the recipient's own concept: two pairs can share a concept, and a
    "control" that accidentally restates the real link would be scored as annotator error.
    """
    if len(pairs) < 2:
        raise ValueError(f"choose_controls: need at least 2 pairs to mispair, got {len(pairs)}")

    rng = random.Random(seed)
    chosen = rng.sample(pairs, k=min(n, len(pairs)))
    controls = []
    for pair in chosen:
        donors = [other for other in pairs if other.concept_id != pair.concept_id]
        donor = rng.choice(donors)
        controls.append(
            replace(
                pair,
                concept_id=donor.concept_id,
                concept_name=donor.concept_name,
                aliases=donor.aliases,
                type_violation=False,
            )
        )
    return controls


def build_rows(
    pairs: list[PairEvidence],
    *,
    controls: list[PairEvidence] | None = None,
    rng: random.Random | None = None,
) -> list[AcronymRow]:
    """Interleave the real pairs with the controls, THEN number them.

    ⚠️ ORDER MATTERS AND IS PINNED BY A TEST. Controls are appended after the real pairs, so
    numbering before the shuffle stamps every control into the trailing block of `row_id`s and
    an annotator reaching the end of the sheet would recognise them on sight. This exact defect
    shipped in the relevance export's first draft.
    """
    flagged = [(pair, False) for pair in pairs] + [(pair, True) for pair in controls or ()]
    for pair, _ in flagged:
        if not pair.aliases and pair.concept_name.strip().lower() == pair.surface.lower():
            raise ValueError(
                f"build_rows: {pair.surface!r} -> {pair.concept_id} has no name beyond the "
                "surface itself and no aliases, so the row asks whether the acronym means the "
                "acronym. Refusing rather than degrading: this would reach the annotator as "
                "`cant_tell` and be read as a Gate 1 tractability problem."
            )
    if rng is not None:
        rng.shuffle(flagged)

    return [
        AcronymRow(
            row_id=f"a{i:03d}",
            surface=pair.surface,
            concept_id=pair.concept_id,
            concept_name=pair.concept_name,
            aliases=pair.aliases,
            contexts=pair.contexts,
            n_mentions=pair.n_mentions,
            type_violation=pair.type_violation,
            is_control=is_control,
        )
        for i, (pair, is_control) in enumerate(flagged)
    ]


#: Aliases printed per row. Some concepts carry 20+ and the wall buries the concept name.
_MAX_ALIASES = 8

#: At or below this length an alias is itself acronym-shaped.
_ACRONYM_LEN = 5


def _shown_aliases(aliases: Sequence[str]) -> list[str]:
    """Spelled-out names first, then acronym-shaped ones, capped.

    An acronym-shaped alias tells the annotator nothing the surface has not already told
    them -- `FXS` listing `fxs` is noise, `martin-bell syndrome` can settle the row. A plain
    alphabetical truncation would systematically keep the noise and drop the signal, since
    short forms sort early as often as not. Order within each group is preserved, and the full
    list stays in the alias artifact.
    """
    spelled = [alias for alias in aliases if len(alias) > _ACRONYM_LEN]
    short = [alias for alias in aliases if len(alias) <= _ACRONYM_LEN]
    return (spelled + short)[:_MAX_ALIASES]


def render_markdown(rows: Sequence[AcronymRow], *, rows_hash: str, seed: int) -> str:
    """The annotator-facing sheet, in the exact block format `parse_annotations` reads.

    ⛔ THREE THINGS `AcronymRow` CARRIES AND THIS NEVER PRINTS.

    **The type-violation flag**, and this is the one that would quietly invalidate §6. Marking
    the pairs whose concept type already contradicts the mention's NER label would have the
    annotator label those rows on the flag rather than on the text, and Gate 4 would then be
    reading the flag's agreement with labels the flag produced.

    **The mention count.** Frequency is not evidence about whether a link is right. `APT` is at
    once the most frequent pair, at 26 of the 204 mentions, and among the clearest errors -- so
    showing it invites an inference that lands on the right answer for the wrong reason, and
    would land on the wrong answer elsewhere.

    **Which rows are controls.** A control that is recognisable measures nothing.

    ⚠️ The context windows are printed verbatim and UNMARKED. 35 of the 44 pairs contain a
    parenthetical gloss and a regex can find them, but highlighting it would make the label a
    judgment about that regex's output, and would split the sheet into rows where the tool
    worked and rows where it did not.
    """
    lines = [
        "# Acronym links — blind adjudication",
        "",
        f"Rows hash {rows_hash}, seed {seed}, {len(rows)} rows.",
        "",
        "Each row shows a **surface form** as it appeared in some abstracts, the **concept "
        "the linker chose for it**, and one or two passages where it appeared. Judge whether "
        "that concept is what the surface means in those passages, and record one of:",
        "",
        "- `correct` — the concept is what the author meant.",
        "- `wrong` — the concept is not what the author meant.",
        "- `granularity` — the right subject, at the wrong level: a parent concept standing "
        "in for the specific thing the text names, or the reverse.",
        "- `cant_tell` — the passages do not settle what the surface means here.",
        "",
        "Add a one-line reason. Judge each row on its own; do not go back and revise earlier "
        "rows once later ones clarify the distinctions, because that turns a blind pass into "
        "a calibrated one.",
        "",
        "Nothing on this sheet encodes the answer: every row has the same shape, whatever "
        "the link is.",
        "",
        "---",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines += [
            f"## {index}. `{row.row_id}`",
            "",
            f"**Surface:** `{row.surface}`",
            "",
            f"**Linked concept:** {row.concept_name}",
            "",
            f"**Also known as:** {', '.join(_shown_aliases(row.aliases))}",
            "",
            "**Where it appeared:**",
            "",
        ]
        lines += [f"> {context}" + "\n" for context in row.contexts]
        lines += ["```", "label:", "reason:", "```", "", "---", ""]
    return "\n".join(lines)


def rows_hash(rows: Sequence[AcronymRow]) -> str:
    """Content-addressed over the frozen row set, stable across order.

    Covers what was SHOWN -- surface, concept, contexts -- and the control marking, so a sheet
    regenerated after labelling began cannot be joined to those labels while claiming the hash
    they were registered against. The mention count and the type-violation flag are excluded:
    they never reach the annotator, and a correction to either must not invalidate a reading
    that was made without them.
    """
    payload = json.dumps(
        sorted(
            (row.row_id, row.surface, row.concept_id, list(row.contexts), row.is_control)
            for row in rows
        ),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


DEFAULT_STATES = "data/synth/states"
DEFAULT_OUT = "data/acronym"
SEED = 20260906
N_CONTROLS = 8


def main(argv: list[str] | None = None) -> None:
    """Freeze the adjudication rows. No network, no model, no linker run.

    `main()` gets no direct unit test per project convention; every function it calls is
    tested, and what is left here is argument plumbing and file IO.
    """
    import argparse
    import collections
    import gzip
    import pathlib

    parser = argparse.ArgumentParser(description="Freeze the DEF-0001 adjudication sheet.")
    parser.add_argument("--states", default=DEFAULT_STATES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--labels", default="data/canon/concept_labels.json.gz")
    parser.add_argument("--aliases", default="data/canon/mesh_aliases.json.gz")
    args = parser.parse_args(argv)

    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(pathlib.Path(args.states).glob("*.json"))
    ]
    with gzip.open(args.labels, "rt", encoding="utf-8") as fh:
        concept_labels = json.load(fh)
    with gzip.open(args.aliases, "rt", encoding="utf-8") as fh:
        alias_table = json.load(fh)

    # The shipped table is surface -> [(concept, name, exact)]; the sheet needs the inverse.
    by_concept: dict[str, set[str]] = collections.defaultdict(set)
    for surface, entries in alias_table.items():
        for concept_id, _name, _exact in entries:
            by_concept[concept_id].add(surface)
    aliases = {cid: tuple(sorted(surfaces)) for cid, surfaces in by_concept.items()}

    pairs = gather_pairs(states, concept_labels=concept_labels, aliases=aliases)
    controls = choose_controls(pairs, n=N_CONTROLS, seed=SEED)
    rows = build_rows(pairs, controls=controls, rng=random.Random(SEED))
    digest = rows_hash(rows)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rows_hash": digest,
        "seed": SEED,
        "n_pairs": len(pairs),
        "n_controls": len(controls),
        "n_mentions": sum(pair.n_mentions for pair in pairs),
        "rows": {
            row.row_id: {
                "surface": row.surface,
                "concept_id": row.concept_id,
                "concept_name": row.concept_name,
                "n_mentions": row.n_mentions,
                "type_violation": row.type_violation,
                "is_control": row.is_control,
            }
            for row in rows
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "sheet.md").write_text(
        render_markdown(rows, rows_hash=digest, seed=SEED), encoding="utf-8"
    )
    flagged = sum(1 for pair in pairs if pair.type_violation)
    print(
        f"{len(pairs)} pairs ({manifest['n_mentions']} mentions), {len(controls)} controls, "
        f"{flagged} type-violating -> {out}/sheet.md  [rows_hash {digest}]"
    )


if __name__ == "__main__":
    main()
