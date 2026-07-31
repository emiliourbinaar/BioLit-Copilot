from collections.abc import Iterable, Sequence
from typing import Protocol

from biolit.domain.paper import Paper
from biolit.domain.records import Entity, ExtractedRecord, Finding
from biolit.ner.windowing import sentence_spans


class Extractor(Protocol):
    """Turns one paper into its finding-bearing sentences.

    Injected keyword-only into `build_record`, mirroring the `Linker` and `PairingStrategy`
    seams: a different extractor later becomes a constructor argument, not a rewrite.
    """

    def findings(self, paper: Paper) -> list[Finding]: ...


def findings_from_sentence_indices(text: str, indices: Iterable[int]) -> list[Finding]:
    """Convert sentence indices into located Findings, DROPPING out-of-range indices.

    This is the whole reason extractors return indices rather than character offsets: an
    index either addresses a sentence or it does not, so a fabricated span is structurally
    impossible rather than merely detected. Out-of-range indices are dropped -- never
    clamped, never guessed -- consistent with every other fail-closed decision here.
    """
    spans = sentence_spans(text)
    out: list[Finding] = []
    for index in sorted(set(indices)):
        if not 0 <= index < len(spans):
            continue
        start, end = spans[index]
        out.append(Finding(text=text[start:end], start=start, end=end, sentence_index=index))
    return out


def build_record(
    paper: Paper, *, entities: Sequence[Entity], extractor: Extractor
) -> ExtractedRecord | None:
    """Assemble one paper's record, or None when its licence forbids extraction.

    THE SINGLE LICENCE ENFORCEMENT POINT, and it suppresses the WHOLE record rather than
    just the findings. ExtractedRecord is not text-free: Entity.text and Finding.text both
    carry verbatim abstract substrings, so emitting entities for a non-extractable paper
    would leak its text downstream even with zero findings.

    Sufficient as the only gate because `Paper` appears in exactly two contracts --
    RetrieverOutput and ExtractorInput -- so the Extractor is the last node that ever sees
    one. Critic and Synthesis take only ExtractedRecord / Cluster / ContradictionFinding.

    Distinct from a safety refusal, which empties key_findings ONLY: a refusal is one LLM
    call declining and has no bearing on entities from the separate NER/linking stage.
    """
    if not paper.extraction_allowed:
        return None
    return ExtractedRecord(
        paper_id=paper.id, entities=list(entities), key_findings=extractor.findings(paper)
    )
