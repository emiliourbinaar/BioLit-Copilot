"""Derive paper-pair contradiction candidates from the Alamri & Stevenson claim corpus.

See `docs/superpowers/specs/2026-09-02-alamri-pi-annotation-design.md`. The corpus records,
for each of 24 systematic-review clinical questions, a set of claims extracted verbatim from
the included papers and assigned `YS` (the paper answers the question yes) or `NO` (it
answers no) by two annotators at 97% agreement. Two papers taking opposite assertions on one
question are a CONTRADICTION CANDIDATE.

That derivation is a PROXY and this module does not assume it is sound. Whether a `YS x NO`
pair is a genuine disagreement, rather than two compatible findings separated by population
or dose, is measured by blind human annotation -- the same instrument that retired the CTD
proxy at pi-hat 0.067 (ADR-0017). Nothing here scores anything; it only builds the manifest
that annotation runs against.
"""

import hashlib
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from biolit.domain.records import ContradictionLabel


@dataclass(frozen=True)
class AlamriClaim:
    """One `<CLAIM>` row: a sentence lifted from `pmid`'s abstract, and the answer the
    original annotators judged it to give to `question`."""

    pmid: str
    question: str
    assertion: str
    text: str


def parse_corpus(xml_text: str) -> list[AlamriClaim]:
    """Flatten the `<CORPUS><REVIEW><CLAIM>` tree into claims.

    The review level is dropped deliberately: `QUESTION` is carried on every `<CLAIM>` and no
    question spans more than one review (measured: 0 of 24), so the question is a complete
    grouping key on its own and the review element adds nothing a pair needs. Review title
    and PMID are not retained at all -- they are withheld from the annotator (titles often
    state the answer outright), and a field that is never read cannot leak.
    """
    root = ET.fromstring(xml_text)
    return [
        AlamriClaim(
            pmid=claim.get("PMID", ""),
            question=claim.get("QUESTION", ""),
            assertion=claim.get("ASSERTION", ""),
            text=(claim.text or "").strip(),
        )
        for review in root.findall("REVIEW")
        for claim in review.findall("CLAIM")
    ]


@dataclass(frozen=True)
class AlamriPair:
    """One derived paper-pair candidate: the unit `ContradictionFinding` is defined over."""

    pair_id: str
    question_id_a: str
    question_id_b: str
    paper_id_a: str
    paper_id_b: str
    label: ContradictionLabel
    stratum: str
    signals: tuple[str, ...]


def pair_id(question_id_a: str, question_id_b: str, paper_id_a: str, paper_id_b: str) -> str:
    """An OPAQUE id for a pair: a digest of the two questions and the two papers.

    Opacity is a blind-protocol requirement, not tidiness. The annotator reads `pair_id` in
    the sheet heading, and a readable id built from its parts would be a structural tell: a
    distractor joins two DIFFERENT questions, so any id carrying both question ids is
    visibly longer or visibly doubled compared with a within-question pair. This project's
    sole annotator is also its designer and knows five distractors exist, so an identifiable
    distractor could be answered `insufficient_overlap` from its id alone -- which is exactly
    the reading the distractors were added to obtain honestly.

    The digest still covers the QUESTIONS, so the three real paper pairs that appear under
    two questions each keep two distinct ids rather than colliding into one manifest row.
    """
    payload = "|".join((question_id_a, question_id_b, paper_id_a, paper_id_b))
    return "p" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def question_id(question: str) -> str:
    """A short stable id for a question, derived from its text rather than its position.

    Position would be stable only for as long as the corpus file is; a content hash survives
    a re-download and makes a manifest row self-describing.
    """
    return "q" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]


#: Words carried by so many of the 24 questions that matching one says nothing about whether
#: a claim addresses its question. Deliberately small and hand-checked rather than a generic
#: English stoplist: the questions are formulaic ("In patients with X, does treatment with Y,
#: compared to Z, reduce W"), so it is the FRAME words that need removing, not common English.
_QUESTION_STOPWORDS = frozenset(
    """in with does do is are the a an of and or to compared for patients women men study
    not no associated risk developing development treatment than been have has""".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")


def key_terms(question: str) -> frozenset[str]:
    """The content words of a question: what a claim must mention to be on-topic for it.

    Tokens shorter than four characters are dropped along with `_QUESTION_STOPWORDS`, which
    removes the numerals and unit fragments that would otherwise match almost any abstract.
    """
    return frozenset(
        word for word in _WORD.findall(question.lower()) if word not in _QUESTION_STOPWORDS
    )


#: Population qualifiers. A `YS x NO` pair whose claims name DIFFERENT populations is the
#: shape that retired the CTD proxy -- two compatible findings that disagree only because
#: they were measured in different people. Matching is deliberately shallow; this is a
#: screening signal whose usefulness is itself what the annotation pass measures, not a
#: validated classifier.
_POPULATION = re.compile(
    r"\b(korean|chinese|japanese|caucasian|african|american|european|indian|black|white|"
    r"asian|hispanic|turkish|iranian|brazilian|australian|men|women|male|female|elderly|"
    r"older|children|postmenopausal|population|ethnic\w*|cohort of)\b",
    re.I,
)

POPULATION_MISMATCH = "population_mismatch"
OFF_QUESTION = "off_question"

#: The four strata. `flagged` and `clean` partition the contradiction candidates and each
#: gets its OWN Gate 2 at n=15; `agreement` and `distractor` are controls that read annotator
#: strictness rather than pi (see `alamri_annotation_export`).
STRATUM_FLAGGED = "flagged"
STRATUM_CLEAN = "clean"
STRATUM_AGREEMENT = "agreement"
STRATUM_DISTRACTOR = "distractor"


def pair_signals(question: str, text_a: str, text_b: str) -> tuple[str, ...]:
    """Which lexical hazard signals fire for a claim pair, in a stable order.

    `off_question` fires when either claim shares no key term with the question it was
    assigned to -- the claim may be about something the question does not ask. `population
    _mismatch` fires when the two claims name different population qualifiers, including the
    case where only one names any.

    A pair is FLAGGED if either fires and CLEAN if neither does. Whether that split is an
    actionable filter is exactly what stratifying pi-hat over it is for; the signals are not
    assumed to work, and a signal that turns out not to predict anything is a result.
    """
    terms = key_terms(question)
    lowered_a, lowered_b = text_a.lower(), text_b.lower()
    signals: list[str] = []

    on_topic_a = any(term in lowered_a for term in terms)
    on_topic_b = any(term in lowered_b for term in terms)
    if not (on_topic_a and on_topic_b):
        signals.append(OFF_QUESTION)

    populations_a = {m.group(0).lower() for m in _POPULATION.finditer(text_a)}
    populations_b = {m.group(0).lower() for m in _POPULATION.finditer(text_b)}
    if (populations_a or populations_b) and populations_a != populations_b:
        signals.append(POPULATION_MISMATCH)

    return tuple(signals)


def build_pairs(claims: Sequence[AlamriClaim]) -> list[AlamriPair]:
    """Cross every `YS` claim against every `NO` claim within each question.

    `pair_id` carries the QUESTION as well as the two PMIDs. Three paper pairs in the real
    corpus appear under two questions each, so an id built from the PMIDs alone would collide
    and silently merge two distinct judgments into one manifest row.
    """
    by_question: dict[str, list[AlamriClaim]] = defaultdict(list)
    for claim in claims:
        by_question[claim.question].append(claim)

    out: list[AlamriPair] = []
    for question in sorted(by_question):
        qid = question_id(question)
        claims_here = by_question[question]
        ys = [c for c in claims_here if c.assertion == "YS"]
        no = [c for c in claims_here if c.assertion == "NO"]
        opposed = itertools.product(ys, no)
        aligned = itertools.chain(itertools.combinations(ys, 2), itertools.combinations(no, 2))
        for pairing, label in (
            (opposed, ContradictionLabel.contradiction),
            (aligned, ContradictionLabel.agreement),
        ):
            for a, b in pairing:
                if a.pmid == b.pmid:
                    continue
                signals = pair_signals(question, a.text, b.text)
                out.append(
                    AlamriPair(
                        pair_id=pair_id(qid, qid, a.pmid, b.pmid),
                        question_id_a=qid,
                        question_id_b=qid,
                        paper_id_a=a.pmid,
                        paper_id_b=b.pmid,
                        label=label,
                        stratum=(
                            STRATUM_AGREEMENT
                            if label is ContradictionLabel.agreement
                            else (STRATUM_FLAGGED if signals else STRATUM_CLEAN)
                        ),
                        signals=signals,
                    )
                )
    return out


def build_distractors(claims: Sequence[AlamriClaim]) -> list[AlamriPair]:
    """Pair papers across questions that share no key term: the known-unrelated anchor.

    These exist to separate the two readings a single annotator cannot otherwise tell apart
    (ADR-0017 defect (b)). If the annotator marks these `insufficient_overlap` but marks the
    within-question `agreement` controls `agreement`, the label is being used for its meaning
    and the pi-hat readings stand. If BOTH come back `insufficient_overlap`, the pull is
    general strictness and pi-hat is confounded in the same direction.

    The key-term-disjointness requirement is what makes the anchor an anchor: two questions
    that share vocabulary could produce a pair an annotator would reasonably call related,
    at which point the row measures nothing.
    """
    by_question: dict[str, list[AlamriClaim]] = defaultdict(list)
    for claim in claims:
        by_question[claim.question].append(claim)

    questions = sorted(by_question)
    terms = {question: key_terms(question) for question in questions}

    out: list[AlamriPair] = []
    for question_a, question_b in itertools.combinations(questions, 2):
        if terms[question_a] & terms[question_b]:
            continue
        qid_a, qid_b = question_id(question_a), question_id(question_b)
        for a, b in itertools.product(by_question[question_a], by_question[question_b]):
            if a.pmid == b.pmid:
                continue
            out.append(
                AlamriPair(
                    pair_id=pair_id(qid_a, qid_b, a.pmid, b.pmid),
                    question_id_a=qid_a,
                    question_id_b=qid_b,
                    paper_id_a=a.pmid,
                    paper_id_b=b.pmid,
                    label=ContradictionLabel.insufficient_overlap,
                    stratum=STRATUM_DISTRACTOR,
                    signals=(),
                )
            )
    return out


#: Draw order. Flagged and clean go first because their quotas are the ones Gate 2 refuses to
#: run without: if the caps bind, the shortfall must land on a control stratum, never on a
#: stratum whose n is fixed at 15 by calibration.
STRATA_ORDER = (STRATUM_FLAGGED, STRATUM_CLEAN, STRATUM_AGREEMENT, STRATUM_DISTRACTOR)

#: The 45-pair batch of the design. Flagged and clean are 15 apiece because `evaluate_gate2`
#: refuses any other n -- its bands are literal binomial-tail counts at n=15, not a
#: proportional rule -- so the sample is shaped to fit the calibrated instrument.
DEFAULT_QUOTAS = {
    STRATUM_FLAGGED: 15,
    STRATUM_CLEAN: 15,
    STRATUM_AGREEMENT: 10,
    STRATUM_DISTRACTOR: 5,
}


def sample_batch(
    pairs: Sequence[AlamriPair],
    *,
    rng: random.Random,
    quotas: Mapping[str, int] = DEFAULT_QUOTAS,
    cap_paper: int = 1,
    cap_question: int = 3,
) -> list[AlamriPair]:
    """Draw the stratified annotation batch under shared independence caps.

    `cap_paper` bounds how many pairs any one paper may appear in ACROSS THE WHOLE BATCH, and
    `cap_question` does the same for a question; a distractor spans two questions and so
    consumes a slot in each. The counters are shared between strata rather than reset per
    stratum, so the bound is on the batch a human actually reads.

    This is the response to the corpus's non-independence: 254 papers carry 728 contradiction
    candidates, median 4 appearances and max 22, so an uncapped draw could let a handful of
    heavily-reused papers dominate. At the design's cap of 1 every paper in the batch appears
    exactly once. Full independence is not achievable in a corpus this size and is not
    claimed; the caps bound the damage rather than removing it.

    RAISES rather than returning a short stratum. `evaluate_gate2` refuses any n != 15, so an
    under-filled contradiction stratum would fail loudly later anyway -- but the control
    strata have no such guard, and a silently-short `agreement` stratum would weaken the
    strictness read with nothing to signal it had happened.
    """
    by_stratum: dict[str, list[AlamriPair]] = defaultdict(list)
    for pair in pairs:
        by_stratum[pair.stratum].append(pair)

    paper_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    batch: list[AlamriPair] = []

    for stratum in STRATA_ORDER:
        wanted = quotas.get(stratum, 0)
        if not wanted:
            continue
        candidates = sorted(by_stratum.get(stratum, []), key=lambda p: p.pair_id)
        rng.shuffle(candidates)

        taken = 0
        for pair in candidates:
            papers = (pair.paper_id_a, pair.paper_id_b)
            questions = {pair.question_id_a, pair.question_id_b}
            if any(paper_counts[p] >= cap_paper for p in papers):
                continue
            if any(question_counts[q] >= cap_question for q in questions):
                continue
            batch.append(pair)
            paper_counts.update(papers)
            question_counts.update(questions)
            taken += 1
            if taken == wanted:
                break

        if taken < wanted:
            raise ValueError(
                f"sample_batch: stratum {stratum!r} yielded {taken} of {wanted} pairs under "
                f"cap_paper={cap_paper}, cap_question={cap_question}. Widen a cap or shrink "
                "the quota -- do not accept a short stratum, since Gate 2 is calibrated at "
                "exactly n=15 and the control strata have no such guard."
            )
    return batch


def write_manifest(pairs: Sequence[AlamriPair], path: str | Path) -> None:
    """Ids, labels, strata and signals only -- never claim or abstract text.

    This file IS committed, unlike the annotation sheet: it carries no licensed corpus text,
    and without it the sample is not reproducible from the seed alone.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(asdict(pair)) + "\n")


def read_manifest(path: str | Path) -> list[AlamriPair]:
    """`label` is coerced back to `ContradictionLabel` and `signals` back to a tuple.

    JSON round-trips a StrEnum to a plain `str` and a tuple to a `list`. Both compare `==`
    to the original in some contexts and neither is `is`-identical, so a manifest read from
    disk would silently differ from one built in memory -- and the strata are derived from
    `label is ContradictionLabel.agreement`.
    """
    out: list[AlamriPair] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            record["label"] = ContradictionLabel(record["label"])
            record["signals"] = tuple(record["signals"])
            out.append(AlamriPair(**record))
    return out


def manifest_hash(pairs: Sequence[AlamriPair]) -> str:
    """Stable over content, not over file bytes, so a re-serialization cannot change it.

    `sort_keys=True` is load-bearing for the same reason it is in `contradiction_gold`:
    without it the hash tracks `AlamriPair`'s field-declaration order, so reordering the
    dataclass would invalidate the corpus pin in every historical run-log line.
    """
    payload = json.dumps([asdict(p) for p in pairs], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
