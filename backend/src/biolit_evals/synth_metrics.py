"""Deterministic metrics for Synthesis Gate A.

Every function here scores an output STRING against the cluster it was generated from. No
gold standard, no annotation, no network: a claim is checkable because the source text is
right there. That is the property the Critic's task lacked and the reason Gate A can be run
in a day rather than over two annotation rounds (spec of 2026-09-03, §1).

THE SUITE IS DELIBERATELY ASYMMETRIC. Support and coverage are DISQUALIFIERS for the LLM
arm, not scores: the template scores 1.0 on both by construction, so comparing the arms
there would be a score one arm cannot lose -- ADR-0015's tautology from the other side. Only
compression against `dcr` is comparative.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from biolit.domain.paper import Paper
from biolit.domain.records import Cluster, ExtractedRecord

#: Integers and decimals. Percent signs and units are excluded from the capture so "1.5%"
#: and "1.5" compare equal -- an output that drops the unit has not invented a number.
#:
#: The guard is a LEADING lookbehind, not a pair of word boundaries. What must be excluded
#: is a digit buried inside an identifier -- the "1" in "HbA1c", the "1" in the paper id
#: "p1" -- and those are recognised by what precedes them. A trailing boundary would also
#: reject the unit written flush against the number, which is how doses and durations are
#: normally written: "500mg" would yield nothing and "1.5mg/kg" would yield a phantom "1".
#: The "." in the lookbehind stops a version-like "v1.2.3" from contributing "2.3".
_NUMERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def numerals(text: str) -> list[str]:
    """Every numeral in `text`, in order, with duplicates kept."""
    return _NUMERAL.findall(text)


@dataclass(frozen=True)
class SourceView:
    """Everything an output about this cluster may legitimately draw on."""

    text: str
    #: The finding sentences alone, with no cluster key and no paper metadata. Spec §2 defines
    #: compression against "concatenated source-findings length", so it needs a denominator
    #: that metadata cannot move; `text` is the wider view the support and hallucination checks
    #: are entitled to draw on. Two different questions, two different strings.
    findings_text: str
    numerals: frozenset[str]
    paper_ids: tuple[str, ...]
    n_papers: int


def build_source_view(
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    papers: Mapping[str, Paper],
) -> SourceView:
    """Collect the cluster's finding sentences and paper metadata into one checkable view."""
    parts: list[str] = [cluster.key]
    findings: list[str] = []
    for pid in cluster.paper_ids:
        paper = papers[pid]
        # TITLE IS DELIBERATELY EXCLUDED. Spec §5 gives each arm the year, journal, PMID and
        # finding sentences -- never the title. Counting a title's numerals as "source" would
        # let an arm cite a figure it was never shown and have it scored as supported: a false
        # pass on a disqualifier, which is the score-that-cannot-be-lost shape ADR-0015 exists
        # to catch. If an arm is ever given titles, this list must change in the same commit.
        parts.extend(str(x) for x in (paper.journal, paper.year, paper.pmid) if x)
        if pid in records:
            findings.extend(f.text for f in records[pid].key_findings)
            parts.extend(f.text for f in records[pid].key_findings)
    text = "\n".join(parts)
    return SourceView(
        text=text,
        findings_text="\n".join(findings),
        numerals=frozenset(numerals(text)),
        paper_ids=tuple(cluster.paper_ids),
        n_papers=len(cluster.paper_ids),
    )


@dataclass(frozen=True)
class Support:
    supported: int
    total: int
    unsupported: tuple[str, ...]

    @property
    def rate(self) -> float:
        return 1.0 if self.total == 0 else self.supported / self.total


def support_rate(output: str, source: SourceView) -> Support:
    """Fraction of the output's numerals that the source can account for.

    A numeral is supported if it appears in the source, OR equals the cluster's paper count
    -- the one aggregate an arm may legitimately introduce ("six papers report...").

    ⚠️ THIS METRIC IS STRICT IN A WAY THAT PENALISES EXACTLY WHAT AN LLM IS FOR. A true
    aggregate it cannot verify ("five of six measured HbA1c") counts against the arm.
    Building a test biased against one arm is the same error family as scoring an arm on a
    population it cannot lose on, pointed the other way. The mitigation is that `unsupported`
    is returned ITEMISED, never as a bare rate, so a reader can tell a hallucinated number
    from an unverifiable-but-true one. Do not report `rate` without it.
    """
    found = numerals(output)
    allowed = source.numerals | {str(source.n_papers)}
    unsupported = tuple(n for n in found if n not in allowed)
    return Support(
        supported=len(found) - len(unsupported), total=len(found), unsupported=unsupported
    )


#: Alias word-count cap for the n-gram scan. MeSH aliases run 1-36 words, but 1-6 covers
#: 98.7% of the 551,669 in the artifact (measured 2026-09-03). Scanning to 36 would multiply
#: the lookup cost sixfold to reach chemical names no generated prose will contain. The
#: truncation is a real limitation and is reported rather than hidden.
MAX_ALIAS_WORDS = 6

_WORD_SPLIT = re.compile(r"[^a-z0-9\-]+")


def _alias_words(text: str) -> list[str]:
    """Lowercase `text` and split it into the exact word tokens both sides key on.

    `load_aliases` and `mesh_concepts` MUST both go through this helper, in lockstep, to
    build a lookup key -- that is the whole invariant the alias table depends on. A key
    built any other way (e.g. plain `.lower()`, keeping punctuation) can never be found:
    the scanner's n-gram keys are always plain word tokens rejoined with single spaces, so
    a comma or other punctuation left in the loader's key -- as in the MeSH inverted form
    "Diabetes Mellitus, Type 2", which is the descriptor's own primary alias -- makes that
    concept permanently unreachable from either side. That drift is the defect this helper
    exists to prevent from recurring.
    """
    # Dangling hyphens are stripped (Ruling 28). `_WORD_SPLIT` treats non-ASCII as a
    # separator, so "TNF-α" tokenises to "tnf-" with the hyphen orphaned, and "tnf-" can never
    # match the alias key "tnf": the source reads as never having mentioned TNF, and an arm
    # writing the plain form is reported as INVENTING it -- a false positive on the one hard
    # disqualifier that must be exactly 0. Measured on the frozen corpus: 9 such tokens,
    # among them tnf-, tgf- and nf-.
    #
    # Stripped on BOTH sides, because Ruling 6's lockstep is what keeps the table reachable at
    # all; normalising only the scan side is the very drift that made 37.57% of aliases
    # unmatchable. The cost is measured rather than assumed: 17.87% of alias keys change and
    # 301 new concept collisions appear, but every sampled collision is IUPAC nomenclature
    # ("10 perfluorohexyl decanol", "3 4 dihydroxyflavone") -- the same population
    # MAX_ALIAS_WORDS already truncates, for the same reason: no synthesis prose contains it.
    # Internal hyphens are untouched, so "non-hodgkin" and "il-6" keep their exact keys.
    return [w for w in (w.strip("-") for w in _WORD_SPLIT.split(text.lower())) if w]


def mesh_concepts(text: str, aliases: Mapping[str, str]) -> set[str]:
    """Every MeSH concept id whose alias appears in `text`.

    Word n-grams are looked up in the alias table rather than scanning 551,669 aliases as
    substrings: the n-gram form is O(len(text)) with a hash lookup per gram, and the
    substring form is O(n_aliases) per call.
    """
    words = _alias_words(text)
    found: set[str] = set()
    for start in range(len(words)):
        for size in range(1, MAX_ALIAS_WORDS + 1):
            if start + size > len(words):
                break
            concept = aliases.get(" ".join(words[start : start + size]))
            if concept is not None:
                found.add(concept)
    return found


def hallucinated_concepts(
    output: str, source: SourceView, aliases: Mapping[str, str]
) -> tuple[str, ...]:
    """MeSH concepts named in the output that no source text mentions, sorted for stability.

    ⚠️ SCOPE, STATED SO IT IS NOT OVERREAD. This catches an invented *MeSH-linkable entity*
    -- a drug or disease that is not there. It does NOT catch an invented study design, an
    invented relationship between two real entities, or a real entity attributed to the
    wrong paper. A zero here is not a clean bill of health; it is the absence of one
    specific, dangerous, mechanically-detectable failure.
    """
    return tuple(sorted(mesh_concepts(output, aliases) - mesh_concepts(source.text, aliases)))


@dataclass(frozen=True)
class Coverage:
    covered: int
    n_papers: int
    missing: tuple[str, ...]

    @property
    def rate(self) -> float:
        return 1.0 if self.n_papers == 0 else self.covered / self.n_papers


def _contains_phrase(haystack: Sequence[str], phrase: Sequence[str]) -> bool:
    """Whether `phrase` occurs as a CONTIGUOUS run of tokens inside `haystack`.

    An empty phrase is never present -- there is nothing there to have been cited.
    """
    phrase = tuple(phrase)
    if not phrase:
        return False
    haystack = tuple(haystack)
    n = len(phrase)
    return any(haystack[i : i + n] == phrase for i in range(len(haystack) - n + 1))


def coverage(output: str, cluster: Cluster, papers: Mapping[str, Paper]) -> Coverage:
    """Fraction of the cluster's papers the output carries an identifying reference for.

    Spec §2 defines a content unit's paper reference as PMID / year+journal; this checks a
    paper's own id, its PMID, and its year+journal together. A DISQUALIFIER for the LLM arm,
    not a score: the template covers every paper by construction, so its 1.0 here says
    nothing about its quality.

    An id or PMID is checked as a CONTIGUOUS TOKEN PHRASE, via `_alias_words` and
    `_contains_phrase`, not by single-token equality: `Paper.id` is `doi or pmid`, and
    bioRxiv's client sets `id=doi or title` with no `pmid` at all. A DOI or title routinely
    contains characters `_alias_words` treats as separators, so it tokenises into several
    words -- a single-token check could never credit it, even cited verbatim. Phrase matching
    still isn't a substring test: a paper's id or PMID must appear as its OWN run of tokens,
    not merely inside a longer one -- `'1234567' in 'PMID 12345678'` is true as a substring,
    but `['1234567']` is not a contiguous run of `['pmid', '12345678']`, so a cluster
    containing both PMIDs still scores the shorter one missing against an output that only
    ever named the longer one. Coverage is a disqualifier, so a false "covered" is a
    disqualifier that cannot fire -- the exact shape this harness exists to catch.

    A paper is ALSO covered when its year appears as a token and its journal appears as a
    contiguous token phrase -- spec §5 hands each arm both, so "the 2019 N Engl J Med study"
    is an identification the spec allows. This path fires only when the (year, journal) pair
    is UNIQUE within the cluster, computed from the cluster's own papers, never from the
    output: two or more papers sharing a (year, journal) cannot be told apart by it, so a
    single mention such as "the 2019 NEJM papers" would otherwise cover several papers at
    once -- a disqualifier satisfiable in bulk is a disqualifier that cannot fire, the same
    shape the id/PMID phrase match exists to close. Papers that share a (year, journal) stay
    creditable by id/PMID alone; a paper missing either a year or a journal is simply not
    creditable this way.

    ⚠️ RESIDUAL LIMITATION, IN TWO PARTS. An id, PMID or journal name written flush against
    other word characters, with no separating space or punctuation (e.g. "PMID12345678"),
    tokenises as one longer word and is counted missing even when a human reader would call
    it present. And journal matching is literal: an arm that writes a journal's abbreviation
    ("NEJM") where the stored field says "N Engl J Med" is not credited, because expanding
    abbreviations would need a mapping this project does not have and inventing one would put
    an unmeasured heuristic inside a disqualifier. Both err toward strictness against the LLM
    arm, which is why `missing` is returned ITEMISED and must never be reported as a bare rate
    -- the same caveat `support_rate` already carries. If the pilot shows an arm identifying
    papers by abbreviation, that is a reason to revisit this, and the itemised list is what
    will show it.
    """
    output_words = _alias_words(output)

    # Uniqueness is keyed on the NORMALISED journal, because that is what the match below
    # compares. Keying the raw field instead would let "N Engl J Med" and "N. Engl. J. Med."
    # count as two distinct journals that a single mention nonetheless matches -- reopening
    # the bulk-credit hole this guard exists to close, through the same drift Ruling 6 records.
    year_journal_counts: dict[tuple[int, tuple[str, ...]], int] = {}
    for pid in cluster.paper_ids:
        paper = papers[pid]
        if paper.year is not None and paper.journal is not None:
            key = (paper.year, tuple(_alias_words(paper.journal)))
            year_journal_counts[key] = year_journal_counts.get(key, 0) + 1

    def _identified(pid: str) -> bool:
        if _contains_phrase(output_words, _alias_words(pid)):
            return True
        pmid = papers[pid].pmid
        if pmid is not None and _contains_phrase(output_words, _alias_words(pmid)):
            return True
        paper = papers[pid]
        if paper.year is not None and paper.journal is not None:
            journal_words = _alias_words(paper.journal)
            key = (paper.year, tuple(journal_words))
            if year_journal_counts[key] == 1 and _contains_phrase(output_words, [str(paper.year)]):
                return _contains_phrase(output_words, journal_words)
        return False

    missing = tuple(pid for pid in cluster.paper_ids if not _identified(pid))
    return Coverage(
        covered=len(cluster.paper_ids) - len(missing),
        n_papers=len(cluster.paper_ids),
        missing=missing,
    )


#: Tokens shorter than this carry no distinguishing power in biomedical prose -- they are
#: articles, prepositions and units, present in every paper's findings.
_MIN_TOKEN_LEN = 4


def distinguishing_tokens(
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    *,
    min_token_len: int = _MIN_TOKEN_LEN,
) -> dict[str, frozenset[str]]:
    """{paper_id: tokens appearing in that paper's findings and NO other paper's in the
    cluster}. These are what an output must preserve to have characterised the group rather
    than merely summarised it into one indistinguishable blur.

    Tokenised through `_alias_words`, the same helper `mesh_concepts`/`load_aliases`/
    `coverage` use (Ruling 6) -- one tokenisation in play, not a second inline copy.

    `min_token_len` is a keyword, not the bare `_MIN_TOKEN_LEN` constant (Ruling 16): the
    per-cluster set-difference already removes anything shared between papers, so the
    floor's stated stopword rationale is redundant, and its live effect is discarding short
    biomedical markers -- measured, "IL6 levels increased..." / "TNF levels decreased..."
    already yield unique sets without any floor at all; a floor of 4 then deletes 'il6' and
    'tnf', the two most distinguishing tokens present. Lowering it is not obviously right
    either, so the pre-registered default stays 4 here; this parameter only makes the
    comparison at a lower value possible.
    """
    per_paper: dict[str, frozenset[str]] = {}
    for pid in cluster.paper_ids:
        findings = records[pid].key_findings if pid in records else []
        text = " ".join(f.text for f in findings)
        per_paper[pid] = frozenset(w for w in _alias_words(text) if len(w) >= min_token_len)
    return {
        pid: tokens - frozenset().union(*(per_paper[o] for o in per_paper if o != pid))
        if len(per_paper) > 1
        else tokens
        for pid, tokens in per_paper.items()
    }


@dataclass(frozen=True)
class DCR:
    retained: int
    scorable: int
    indistinguishable: tuple[str, ...]
    no_findings: tuple[str, ...]
    lost: tuple[str, ...]

    @property
    def rate(self) -> float:
        return 1.0 if self.scorable == 0 else self.retained / self.scorable


def dcr(
    output: str,
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    *,
    min_token_len: int = _MIN_TOKEN_LEN,
) -> DCR:
    """Fraction of scorable papers whose distinguishing content survives into the output.

    `min_token_len` is threaded straight through to `distinguishing_tokens` (Ruling 16);
    the default is unchanged, so default behaviour is unaffected -- this only makes the
    floor settable so a sensitivity figure at a lower value can be logged elsewhere.

    THE ONLY COMPARATIVE AXIS, alongside compression. A paper with no distinguishing tokens
    is excluded from the denominator and reported in one of two DISTINCT categories, because
    they mean opposite things for the pilot report (Ruling 10): `indistinguishable` is a
    paper that HAS findings identical to another paper's in its cluster -- a property of the
    CORPUS, no output could tell those two apart -- while `no_findings` is a paper for which
    extraction produced nothing at all -- a property of EXTRACTION, not of the corpus.
    Conflating them would make the report say "the corpus carries duplicates" when the real
    story is "extraction failed on this paper".

    Retention is TOKEN membership, not substring (Ruling 9): `tokens[pid]` is intersected
    with `set(_alias_words(output))`, using the same helper `distinguishing_tokens` builds
    its side from. A substring test (`token in output.lower()`) would count a distinguishing
    token retained whenever any longer word in the output happens to contain it -- 'cyst' in
    "Polycystic ovary morphology was unchanged", 'oral' in "Temporal trends were flat" --
    and the bias runs TOWARD the LLM arm, which writes more and longer prose than the
    template: exactly the false "LLM wins" ADR-0015 exists to catch.

    ⚠️ AND A SECOND BIAS, RUNNING THE OTHER WAY, PRE-REGISTERED IN SPEC §2 ON 2026-09-04
    BEFORE ANY ARM WAS RUN. Token-exact retention REWARDS COPYING. The template reproduces
    finding sentences nearly verbatim and cannot lose; a faithful paraphrase can preserve
    every fact and still score lower. On "Metformin reduced hirsutism scores." and
    "Metformin reduced ovulation latency.", the template scores retained=2/2 while a
    paraphrase carrying both facts scores retained=1 with lost=('p1',). This one favours the
    arm the gate already defaults to shipping, so it must not be waved through as merely
    conservative. Mitigation is reporting: `lost` is per paper, never only a rate, and a DCR
    gap made of genuinely dropped content is a different result from one made of paraphrase.
    Treating a DCR difference as decisive without reading `lost` is not supported here.
    """
    tokens = distinguishing_tokens(cluster, records, min_token_len=min_token_len)
    output_tokens = set(_alias_words(output))
    no_findings = tuple(
        pid
        for pid in cluster.paper_ids
        if not (records[pid].key_findings if pid in records else [])
    )
    indistinguishable = tuple(pid for pid, t in tokens.items() if not t and pid not in no_findings)
    scorable = [pid for pid, t in tokens.items() if t]
    lost = tuple(pid for pid in scorable if not (tokens[pid] & output_tokens))
    return DCR(
        retained=len(scorable) - len(lost),
        scorable=len(scorable),
        indistinguishable=indistinguishable,
        no_findings=no_findings,
        lost=lost,
    )


#: Vocabulary that asserts an epistemic relation between papers. ADR-0018 established the
#: pipeline cannot support such a claim, so its presence is a COMPLIANCE signal.
#:
#: STEM PREFIXES, not surface forms (Ruling 23, superseding Ruling 14). Ruling 14's own
#: docstring claimed the exact-token list "covers inflections of every stem, not only its
#: dictionary form" -- false as shipped, and demonstrated: "corroborated" and "refuting" are
#: missing inflections of stems already in that list, "consistency" was absent while
#: "inconsistency" was present, and the whole "concur" family was missing. 39 hand-enumerated
#: tokens was already the second attempt at this; a hand-maintained inflection list keeps
#: losing this game, so a word now matches if it STARTS WITH one of these stems, which makes
#: the coverage claim true by construction instead of by vigilance.
#:
#: Two stems are deliberately narrower than they look, and that narrowness is the design:
#: "consisten", not "consist", so "the cohort consists of 40 patients" does not fire; and
#: "concur" carries an explicit exclusion (`JUDGMENT_EXCLUSIONS`) for "concurrent" and
#: "concurrently", which mean "at the same time" and are ordinary biomedical vocabulary, not
#: an agreement claim. Deliberately excludes generic words ("similar", "different", "differ")
#: that fire on ordinary prose unrelated to an agreement claim.
JUDGMENT_STEMS = frozenset(
    {
        "agree",
        "disagree",
        "conflict",
        "consisten",
        "inconsisten",
        "discrepant",
        "mixed",
        "opposing",
        "concordan",
        "discordan",
        "corroborat",
        "refut",
        "consensus",
        "divergen",
        "contradict",
        "concur",
    }
)

#: Words that would otherwise match a `JUDGMENT_STEMS` prefix but do not carry the judgment
#: meaning: "concurrent"/"concurrently" start with "concur" but mean "at the same time".
JUDGMENT_EXCLUSIONS = frozenset({"concurrent", "concurrently"})


#: `judgment_language`'s OWN split, deliberately not `_alias_words` (Ruling 23, partly
#: reversing Ruling 15). `_alias_words` keeps '-' as a word character, which is correct for
#: MeSH ("non-hodgkin" must stay one token) but wrong here: "broadly-consistent" tokenised
#: that way is the single word `broadly-consistent`, which matches no judgment vocabulary and
#: never fires. One helper cannot serve both requirements.
_JUDGMENT_WORD_SPLIT = re.compile(r"[^a-z0-9]+")


def _judgment_words(text: str) -> list[str]:
    return [w for w in _JUDGMENT_WORD_SPLIT.split(text.lower()) if w]


def judgment_language(output: str) -> tuple[str, ...]:
    """Judgment vocabulary present in the output, sorted.

    LOGGED, NEVER SCORED. It measures instruction compliance, not characterisation quality,
    and folding it into a threshold would hand the gate a second comparative axis by the
    back door -- which is how ADR-0017's forbidden metric would return under a new name.

    Tokenised through `_judgment_words`, not `_alias_words` -- see that function's docstring.
    Matched by STEM PREFIX against `JUDGMENT_STEMS`, minus `JUDGMENT_EXCLUSIONS` (Ruling 23):
    the terms returned are the actual WORDS found in `output`, not the stems, so a reader
    sees `corroborated`, not `corroborat` -- that itemisation is what spec §5.1's mitigation
    relies on, and stemming must not take it away.
    """
    words = set(_judgment_words(output)) - JUDGMENT_EXCLUSIONS
    matched = {w for w in words if any(w.startswith(stem) for stem in JUDGMENT_STEMS)}
    return tuple(sorted(matched))


def judgment_volunteered(output: str, source: SourceView) -> tuple[str, ...]:
    """Judgment vocabulary the arm introduced, i.e. that the source never gave it.

    LOGGED, NEVER SCORED, exactly as `judgment_language` is. Read the two TOGETHER: neither
    is sufficient alone. Measured on the frozen sample, the template -- which quotes source
    sentences verbatim and cannot volunteer anything -- scored 10/30 clusters on
    `judgment_language`, and all 12 hits were CLINICAL uses in quoted text ("consistent with
    exertional rhabdomyolysis", "mixed high and normal anion gap metabolic acidosis"). So the
    unfiltered count drowns §5.1's signal in the corpus's own vocabulary. But subtracting the
    source is not sufficient either: an arm writing "the findings are consistent across
    studies" over a corpus whose papers say "consistent with rhabdomyolysis" would go
    unflagged here, and that is the exact direction §5.1 exists to catch.

    Compared against `source.text`, not `findings_text`: the question is what the arm was
    GIVEN, and it was given the journal, year and PMID as well as the findings.

    This is construction-true at 0 for the template arm -- its output is boilerplate plus
    verbatim source, and none of its boilerplate ("papers", "PMID", "no finding sentence
    extracted") is judgment vocabulary. A non-zero value there means the template has begun
    generating prose.
    """
    given = set(_judgment_words(source.text))
    return tuple(w for w in judgment_language(output) if w not in given)


def compression(output: str, source: SourceView) -> float:
    """Output length over concatenated source-FINDINGS length (spec §2).

    Below 1.0 means the arm said it shorter than the sentences it was given. The denominator
    is `findings_text`, not `text`: dividing by the whole source view would fold the cluster
    key and every paper's journal, year and PMID into it, so the same findings in a
    long-named journal would score as more compressed than in a short-named one, and the
    statement above would stop being true.

    A cluster with no extracted findings has no denominator and returns 0.0 -- a sentinel,
    not a measurement. `dcr` reports those papers in `no_findings`; read the two together.
    """
    return len(output) / len(source.findings_text) if source.findings_text else 0.0


@dataclass(frozen=True)
class OutputScore:
    support: Support
    coverage: Coverage
    dcr: DCR
    compression: float
    hallucinated: tuple[str, ...]
    judgment_terms: tuple[str, ...]
    #: Ruling 27. Read WITH `judgment_terms`, never instead of it: this one is
    #: construction-true at 0 for the template arm, that one carries the clinical vocabulary
    #: the corpus itself uses. Neither is sufficient alone; see `judgment_volunteered`.
    judgment_volunteered: tuple[str, ...]


def score_output(
    output: str,
    cluster: Cluster,
    records: Mapping[str, ExtractedRecord],
    papers: Mapping[str, Paper],
    aliases: Mapping[str, str],
) -> OutputScore:
    """Run every Gate A metric over one arm's output for one cluster.

    `records[pid]` is guarded (`if pid in records`) inside `build_source_view` and
    `distinguishing_tokens`, while `papers[pid]` is indexed bare here and in `coverage` --
    an asymmetry raised three times in review (Ruling 13) and correct as written:
    `cluster_papers` (biolit/cluster/group.py:29-35) builds `paper_ids` FROM the
    ExtractedRecords, so every id in a cluster came from a record and `records[pid]` cannot
    miss; `papers[pid]` is covered because the runner reads `candidate_papers` and
    `extracted_records` from one pipeline state, so a cluster's paper ids are always a
    subset of both mappings' keys.
    """
    source = build_source_view(cluster, records, papers)
    return OutputScore(
        support=support_rate(output, source),
        coverage=coverage(output, cluster, papers),
        dcr=dcr(output, cluster, records),
        compression=compression(output, source),
        hallucinated=hallucinated_concepts(output, source, aliases),
        judgment_terms=judgment_language(output),
        judgment_volunteered=judgment_volunteered(output, source),
    )


def load_aliases(path: str) -> dict[str, str]:
    """{alias: MeSH id} from the Phase 3A artifact, which is alias-major and prefixes ids.

    Primary aliases win; a non-primary one is kept only if nothing else claims that alias.
    """
    import gzip
    import json

    out: dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for alias, entries in json.load(fh).items():
            key = " ".join(_alias_words(alias))
            if not key:
                continue
            for raw_id, _canonical, is_primary in entries:
                if is_primary or key not in out:
                    out[key] = raw_id.split(":", 1)[1] if ":" in raw_id else raw_id
    return out
