# Defect Log

Measured defects in shipped components that are **recorded and not yet fixed**. Newest at
top. Citable as `DEFECTS.md DEF-000N`.

This log exists because these were being recorded inside whichever ADR happened to notice
them, which buries a standing defect in the prose of a decision about something else. An
entry here is a claim about behaviour that has been *verified against real data*, with the
verification shown. A suspicion is not a defect entry.

**Every entry through DEF-0004 is in entity linking.** That is the project's known dominant
bottleneck (Phase 3: e2e NIL rate 0.45 on the domain sample, 0.32 on BC5CDR), and it is where
a defect does the most damage, because everything downstream — pairing, clustering, selection,
synthesis — consumes `canonical_id` and has no way to second-guess it. **DEF-0005 is the first
entry in the clustering layer**, and it is there because the loss it measures happens *after*
linking succeeds. **DEF-0006 is the first that is not about the system being wrong at all** — it
is about the system disclosing something it had correctly declined to use. **DEF-0007 is
the first found by a GUARD rather than by reading** — six code reviews passed over the code
without seeing it, because it is invisible in any single module and only shows up in
generated data. **DEF-0008 is the first in the retrieval client, and the first found by
rendering data for a human to read** — an attribution list printed DOIs from the wrong journals
and the wrong decades. It is also DEF-0007's actual cause.

---

## DEF-0008 — The PubMed client reads a paper's DOI and PMC id from its reference list: a cited paper's identifiers become the citing paper's own, including the one that decides its licence

- **Date:** 2026-09-10
- **Component:** `biolit.clients.pubmed` — `_parse_article` (the DOI) and `_pmc_id_of` (the PMC
  id, and through it the licence)
- **Status:** **Parser FIXED 2026-09-10; consequences for past corpora NOT yet audited.** Both
  lookups now read only the paper's own `PubmedData/ArticleIdList`, a cassette carrying a real
  `<ReferenceList>` pins that, and the parsing functions are in the fixture pin by AST subtree.
  Regenerated the same day, the four fixtures carry **0** DOIs that are not the paper's own and
  **0** allowed papers without a PMC record of their own, across 237 papers, and `statins`
  no longer collapses any paper. Every other corpus this client has touched is audited
  separately, and no published number changes until that audit reports.
- **Severity:** ⛔ **RIGHTS.** Papers were licensed on **another article's** Creative Commons
  licence and their abstracts quoted verbatim. That is the exact failure the licence gate exists
  to prevent, reached upstream of the gate, where no downstream check could see it. It also
  corrupts attribution (the DOI a reader would cite) and paper identity (`Paper.id`).

### What was observed

Both lookups search the WHOLE `PubmedArticle` element:

```python
for aid in article.findall(".//ArticleIdList/ArticleId"):   # pubmed.py:212
    if aid.get("IdType") == "doi":
        doi = aid.text                                      # last match wins
```

`.//` descends into every `ArticleIdList` under the article, and PubMed's XML carries one per
**cited reference** (`PubmedData/ReferenceList/Reference/ArticleIdList`) after the article's own
(`PubmedData/ArticleIdList`). So:

- **DOI — last match wins.** Any paper whose reference list contains a DOI gets the DOI of the
  **last cited reference that has one**, not its own.
- **PMC id — first match wins** (`_pmc_id_of`, `pubmed.py:50`). Correct when the paper has a PMC
  record, because its own id list comes first. When it has **none**, the first cited reference
  with a PMC id supplies one — and that id is what `_fetch_licences` looks up, so the paper is
  licensed on the **cited article's** licence.

**Measured 2026-09-10 on the 236 papers of the four evidence-viewer fixtures**, each compared
with its own `PubmedData/ArticleIdList` fetched from PubMed:

| | |
|---|---|
| DOI is a **cited paper's**, not the paper's own | **98 of 236 (41.5%)** — every one of the 98 had a DOI of its own |
| no PMC record of its own, given a **cited article's** PMC id | **11** |
| of those 11, **allowed** through the licence gate on the borrowed licence | **10** |
| of those 10, **quoted verbatim** in a committed fixture's answer | **5** |

### Why it matters

- **Rights.** Five abstracts were quoted under a licence that belongs to a different article.
  The fixtures carrying them had been pushed to the `feat/evidence-viewer-python` branch; that
  branch's history was rewritten on 2026-09-10 to remove every fixture version, and `master`
  never carried them. The orphaned commits stay reachable by direct SHA on GitHub until GitHub
  garbage-collects them, which is outside this repository's control.
- **Attribution.** Where a quote was legitimately licensed, the DOI shown for it was a cited
  paper's 41.5% of the time — the attribution the licence requires pointed at the wrong work.
- **Identity.** `Paper.id` is `doi or pmid`, so two papers whose reference lists end with the
  same DOI get the same id and collapse into one. **That is DEF-0007's observed collision**:
  PMIDs 42694066 and 42440952 (own DOIs `10.3389/fphar.2026.1895524` and
  `10.3389/fphar.2026.1852621`) both end their reference lists with `10.3389/fphar.2024.1445324`.
- `full_text_pointer` is built from the same PMC id, so it can point at the wrong article.

### Why nothing caught it, which is the transferable part

**Every test input was trimmed below the shape that triggers it.** None of the five PubMed
cassettes under `tests/cassettes/` contains a `<ReferenceList>`, so `.//` and a direct child
path return the same answer on all of them — the tests pass either way and pin nothing about
which `ArticleIdList` is read. The fixture pin also excluded this module on purpose
(`fixture_pin.py`, gap 1), with the upgrade path written down: *"If it ever bites, hash that
function's AST subtree alone."* It has now bitten.

### What is NOT claimed, yet

The 236-paper figure is from the four fixtures only. **How many papers in every other corpus
this client has touched carry a borrowed DOI or licence — and whether any published number
rests on one — is unmeasured at the time of filing**, and is the next piece of work. No
published number is changed by this entry.

---

## DEF-0007 — Two retrieved papers given the same `Paper.id` silently become one record, and no stage says so: the ledger stops balancing and a paper disappears

- **Date:** 2026-09-09
- **Component:** `biolit.pipeline.stages.records_stage` — the collapse; `biolit.clients.pubmed`
  supplies the colliding id, **by reading it from the papers' reference lists (DEF-0008)**
- **Status:** **Accounting half FIXED 2026-09-09; rights half (the addendum below) NOT fixed.**
  When two retrieved papers are given the same id the second still overwrites the first, but
  `records_stage` now counts the collapse in `dropped` as `duplicate_paper_id`, so the ledger
  balances and the lost paper is visible rather than silent. ⚠️ Corrected 2026-09-10: this
  line read "Recorded, **not fixed**" after the accounting fix had landed, contradicting the
  severity line below it.
- ⛔ **Cause CORRECTED 2026-09-10.** This entry originally said the two papers **shared a DOI**.
  They did not. The collision was manufactured by **DEF-0008**: both papers' reference lists end
  with the same DOI, and the client read that as each paper's own. The ledger arithmetic below
  was real and the accounting fix stands; the explanation of where the shared id came from was
  wrong, and is kept below, struck through by this note, rather than silently rewritten.
- **Severity:** Silent data loss, plus a self-contradicting ledger. Unlike DEF-0001 through
  DEF-0005 this is not a wrong answer — it is a **missing** one that the accounting was supposed
  to make impossible to miss. ⛔ **And see the addendum: the same collision has a second site
  that is a LIVE RIGHTS RISK, not an accounting one** — refused text reaching an allowed paper's
  record. The accounting half is fixed; that half is not.

### What was observed

`Paper.id` is `doi or pmid`. `records_stage` accumulates into a dict keyed on it:

```python
records[paper.id] = record          # stages.py:119
```

When two retrieved papers are given the same id, the second overwrites the first. The
`licence_gate` report is then built from that collapsed dict:

```python
n_in=len(papers), n_out=len(records), dropped=refused   # refusals only
```

so the paper vanishes from `n_out` without being counted anywhere in `dropped`.

**Measured on `statins and rhabdomyolysis`, reproduced on two independent retrievals a day
apart:**

| | |
|---|---|
| retrieved | 58 |
| licence-refused | 17 |
| should survive | **41** |
| records actually built | **40** |
| stage stubs emitted | 57 for 58 retrieved |

⚠️ **`StageReport`'s own docstring promises this cannot happen:** *"Where the two units match,
the ledger is checkable: `n_in - sum(dropped) == n_out`."* On this run it is 58 − 17 = 41 ≠ 40.
The invariant the ledger advertises is false, and nothing in the pipeline noticed.

⛔ **Where the shared id came from — CORRECTED 2026-09-10 (DEF-0008).** The two papers were
PMIDs 42694066 and 42440952, both in *Frontiers in pharmacology*, with distinct DOIs of their
own (`10.3389/fphar.2026.1895524`, `10.3389/fphar.2026.1852621`). Each ends its reference list
with `10.3389/fphar.2024.1445324`, and the client read that cited DOI as each paper's own. Keyed
on the papers' own DOIs, the same retrieval has **no** collision. "Reproduced on two independent
retrievals a day apart" is still true, and is explained by the same two papers being retrieved
both times.

### Why it matters beyond the arithmetic

**A paper that passed the licence gate is dropped from the answer with no record of it.** Every
downstream count — clusters, cited papers, the rendered answer — is computed over 40 papers
while the ledger claims 41 survived. The stage ledger exists precisely so that a reader can
account for every paper; this is the one loss it cannot see.

**It was found by a guard, not by review.** Six independent code reviews passed over the code
without catching it, because it is invisible in any single module: the id policy is in
`clients/`, the collapse is in `stages.py`, and the contradiction is only observable in
generated data. A whole-branch reviewer found it in a committed artifact, and an assertion added
afterwards reproduced it on fresh data.

### What is NOT claimed

The frequency is unmeasured. It was seen twice on one query out of four. ⚠️ **An earlier
version of this paragraph said "duplicate DOIs in PubMed are not rare". That was an assertion,
not a measurement, and the one instance it was explaining turned out not to be a duplicate DOI
at all (DEF-0008).** Whether PubMed ever returns two records carrying the same DOI of their own
is unmeasured; in the 2026-09-10 audit of 236 fixture papers it happened zero times. **The last
write wins**, so which paper survives is
determined by retrieval order rather than by any rule — that is also unexamined, and no claim is
made that keeping the last is better or worse than keeping the first.

### ⛔ ADDENDUM 2026-09-09 — a SECOND collision site: a LIVE RIGHTS RISK IN PRODUCTION, not an accounting bug

⚠️ **Severity, stated plainly because the rest of this entry is about arithmetic and this is
not.** The defect above miscounts. **This one can put a licence-refused paper's verbatim text
inside an allowed paper's record**, which is a rights and attribution failure of the same class
as DEF-0006 — the thing the licence gate exists to make impossible. It is unreachable in the
evidence-viewer fixtures specifically, and that narrowness must not be read as low severity:
`--json-out` is a shipped code path, and the gate's whole design premise is that no refused
paper's text survives `build_record`. Here it does.

Found by a fresh-context review of the accounting fix above. **`records_stage` is not the only
place keyed on `Paper.id`.** One stage earlier, `entities_stage` does the same thing:

```python
by_paper[paper.id] = entities       # stages.py:75
```

The consequence is worse than a miscount. When papers A and B share an id and **B is
licence-refused while A is allowed**, `by_paper[id]` holds **B's** entities, and
`build_record(A, entities=B's)` returns a record — because *A* is allowed. `Entity.text` carries
verbatim substrings of the abstract, and `extract/base.py` states plainly why that matters:
the gate suppresses the whole record precisely because entity text leaks abstract text.

⚠️ **So a refused paper's text can reach an `ExtractedRecord` through an allowed paper's id.**
`collapsed` stays 0, the ledger balances, and nothing fires — the accounting fix above does not
touch this path.

⚠️ **Updated 2026-09-10 for DEF-0008.** While DEF-0008 stands, "sharing an id" requires only that
two papers end their reference lists with the same DOI — far more common than a genuine
duplicate. Once DEF-0008 is fixed, a collision needs two records carrying the same DOI of their
own, whose frequency is unmeasured. That makes this path **latent, not closed**: the key is
still collidable, and the fix for DEF-0008 does not change what happens when it collides.

**Scope, stated precisely.** This is **not** reachable in the evidence-viewer fixtures:
`PaperStub` and `FixtureCluster` carry no entity text, and `Finding.text` is sliced from the
allowed paper's own abstract. It **is** live in `--json-out`, which serialises
`ExtractedRecord.entities` — the same surface as **DEF-0006**, reached through this defect's
mechanism rather than that one's.

**Not fixed, and deliberately not patched in passing.** The accounting fix did not touch
`entities_stage`. Fixing this properly means deciding what `Paper.id` should be — whether a DOI
may serve as a primary key at all when the source might emit it twice, or, as DEF-0008 showed,
when the client can misread it — and that is an
architectural question deserving its own design pass, not a rushed edit inside a frontend
branch. **The risk is recorded here at its real severity so that decision is made deliberately
rather than by default.**

**What would close it, for whoever picks it up:** either a key that cannot collide (PMID-first,
or a composite), or entity storage that is not keyed on `Paper.id` at all. Both are behaviour
changes to the pipeline's identity model and both need their own measurement of what breaks.

### A downstream mislabel, recorded while it is nameable

`biolit_evals/relevance_screen.py` reads `n_licensed = stages["licence_gate"]["n_out"]`. On a
run with a collapse that is the count of *records built*, not papers licensed — 40 where 41 were
licensed. Pre-existing and harmless to that module's yield-only purpose, but the field is
misnamed and the fix above is what makes it possible to say so.

---

## DEF-0006 — `--json-out` serialises the abstracts of papers the licence gate refused: the gate's own justification for being the sole enforcement point lapsed when a second consumer of `Paper` was added

- **Date:** 2026-09-08
- **Component:** `biolit.pipeline.__main__.emit_run` — the enforcement point itself
  (`biolit.extract.base.build_record`) is correct and unchanged
- **Status:** Recorded, **not fixed**. Found while scoping a frontend; independent of it.
- **Severity:** A rights defect, not an accuracy one — the first entry in this log that is not
  about the system being wrong but about it disclosing something it declined to use.

### What was observed

`build_record` refuses a paper whose licence forbids extraction and returns `None`, suppressing
the **whole** record rather than just the findings — because `Entity.text` and `Finding.text`
both carry verbatim abstract substrings. That is correct and still works.

⚠️ **But `PipelineState.candidate_papers` holds every retrieved `Paper`, abstract included,
whether or not the gate refused it** — and `--json-out` serialises the entire state. Measured
on a live `clozapine and agranulocytosis` run at 20 papers:

| | |
|---|---|
| licence tiers present | `unknown` 8, `non_commercial` 5, `open` 7 |
| papers the gate **refused** | **8 of 20** |
| — of those, carrying a **verbatim abstract** in the JSON | **7** |
| example | `license_tier='unknown'`, `license=None`, `abstract_len=2096` |

### Why it happened, which is the transferable part

The gate's docstring states its own sufficiency condition explicitly:

> *"Sufficient as the only gate because `Paper` appears in exactly two contracts —
> RetrieverOutput and ExtractorInput — so the Extractor is the last node that ever sees one."*

⭐ **That was true when it was written, and it silently stopped being true.** `--json-out`
arrived with the end-to-end pipeline and is a **third** consumer of `Paper` — one that reads
`PipelineState` wholesale rather than through a node contract, so it was invisible to the
reasoning above. **The defect is not that the argument was wrong; it is that the argument had a
precondition and nothing checked it when the precondition changed.** A guard asserting that
condition — no serialisation path emits a `Paper` the gate refused — would have failed the day
`--json-out` was added.

### What is NOT claimed

**No artifact containing this has ever been published.** `backend/.gitignore` ignores `data/`
wholesale, so no run state is committed, and the two run logs that *are* committed
(`relevance_runs.jsonl`, `acronym_runs.jsonl`) carry gate readings and hashes, not text. The
exposure is local-only today. It becomes real the moment a run artifact is shipped anywhere —
which is exactly what a frontend fixture would do.

**A second, softer point that this entry does not resolve: extraction permission is not
redistribution permission.** Even the tiers the gate *allows* include `cc_by_nc_nd`, which
permits redistribution with attribution but nothing else. Anything that displays abstract text
publicly needs attribution and probably its own tier policy, and that is a separate decision
from this defect.

---

## DEF-0005 — Same-sentence clustering collapses on class-referring prose: a paper that says "anticoagulation" rather than "warfarin" produces no cluster at all, and the attrition is flat in paper count

- **Date:** 2026-09-08
- **Component:** `biolit.cluster` — `SameSentencePairing`, downstream of linking rather than in it
- **Status:** Recorded, not fixed. Measured on 25 queries × 40 papers, run 2026-09-07/08.

A cluster requires a linked chemical and a linked disease **in the same sentence**. Literature
about a drug *class* refers to the class collectively — "anticoagulation", "DOACs", "PPI
therapy", "immunosuppression" — rather than naming a member, and papers retrieved by a
class-level query lose their chemical–disease pairs at a rate that scales with how collective
the drug term is. The disease side links normally throughout.

⚠️ **THE PROXIMATE CAUSE IS THE SENTENCE BOUNDARY, NOT THE LINKER, and the first draft of this
entry got that wrong.** The obvious story — "the class term NILs, so there is no chemical" — is
not what the records show. On `anticoagulants and intracranial hemorrhage`, **12 of 19 records
carry BOTH a linked chemical and a linked disease, and 0 clusters formed.** The entities exist
and are linked; they never land in one sentence. Compare `tamoxifen and endometrial cancer`,
29 of 30 records with both, 8 clusters.

| query | records | with linked chemical | with linked disease | with **both** | clusters |
|---|---|---|---|---|---|
| anticoagulants / ICH | 19 | 12 | 19 | **12** | **0** |
| immunosuppressants / OI | 19 | 14 | 18 | **14** | **0** |
| PPIs / C. difficile | 30 | 20 | 29 | **20** | **1** |
| tamoxifen / endometrial | 30 | 29 | 30 | **29** | **8** |
| clozapine / agranulocytosis | 20 | 18 | 17 | **17** | **6** |

The collective-terminology component is real but secondary, and it shows up in *which* surfaces
fail to link: the most frequent unlinked chemical surfaces on exactly these queries are the
class abbreviations — **`doac`, `ppis`, `ppi`, `inhibitors`, `mmf`**. So the class term does go
unlinked; the papers simply also contain named agents that link fine, in other sentences,
discussing other things.

**The gradient is monotone in how collective the drug term is**, which is what makes this a
mechanism rather than a set of unlucky queries:

| query drug term | queries | licensed papers | clusters | `no_cluster` |
|---|---|---|---|---|
| single agent (`clozapine`, `tamoxifen`, `vancomycin`) | 5 | 116 | 34 | **31%** |
| structural class (`fluoroquinolones`, `tetracyclines`) | 6 | 117 | 21 | 53% |
| pharmacological action class (`anticoagulants`, `immunosuppressive agents`) | 14 | 295 | 54 | **74%** |

Worst individual cases, all at 40 papers: `anticoagulants and intracranial hemorrhage` **19 of
19 papers dropped, 0 clusters**; `immunosuppressive agents and opportunistic infections` 19 of
19, 0 clusters; `proton pump inhibitors and Clostridioides difficile infection` 28 of 30, 1
cluster. `Anti-Bacterial Agents` — the largest pharmacological class in MeSH at 209 members —
yielded **2 clusters from 15 licensed papers**.

⚠️ **IT IS NOT A SAMPLING PROBLEM, and this is the part that took a measurement to establish.**
Re-running three of the empty queries at 150 papers instead of 40 — 3.75× the literature —
moved cluster counts from 0/1/0 to 7/9/3 while leaving the attrition rate essentially
unchanged: 85%, 86%, 91%. **Yield is linear in papers at a very bad constant**, so buying more
literature buys proportionally more of the same loss rather than escaping it.

**Related to DEF-0001, and deliberately not folded into it.** Both are failures to resolve a
surface to the right concept, but the shape differs and so would any fix. DEF-0001 is *acronym
ambiguity* — a short surface that maps confidently to the wrong concept, producing a **wrong
link**. This is *collective terminology* — a surface naming a class or a therapy rather than an
agent, producing **no link and therefore no pair**. One ships a bad answer; the other ships no
answer, silently, with a completed-looking stage ledger. A context-aware linker might fix
DEF-0001 and do nothing here, because there is often no specific agent in the sentence to
recover.

**Why it was found, and what it cost.** It surfaced while building a fresh label set to
validate ADR-0022's ordering claim, and it is the reason that validation could not be run:
ADR-0022's mechanism fires on exactly the class-referring queries this defect empties. See
**ADR-0023**. The defect is recorded here rather than there because it is independent of
ADR-0022 — it caps recall on any class-level question the system is asked, whether or not
anything is being validated.

⭐ **THIS GIVES ADR-0013's ALTERNATIVE (4) ITS FIRST DEMONSTRATED CONSUMER.** That ADR listed
"loosen same-sentence to same-paragraph or an N-token window" as **"not measured, not
rejected"** — a cheap sweep with no reason to run it, since the 77.0% figure capped what any
pairing tweak could win *on BC5CDR*. The table above is a reason: on class-level queries the
loss is specifically at the sentence boundary, with both endpoints present and linked. ⚠️ That
is a motivation to **measure** the sweep, not evidence it would work — ADR-0013's headroom
argument still stands, and a wider window trades precision for exactly the recall it buys.

⚠️ **A SECOND, SEPARATE THING VISIBLE IN THE SAME DATA, recorded because it was seen rather
than because it was investigated.** The unlinked chemical surfaces include obvious NER span
fragments: **`ofiban`** (from tirofiban), **`tiapine`** (quetiapine), **`zaril`** (Clozaril),
**`oides`** (Clostridioides), **`clo`**, **`do`**, **`tir`**. That is span-boundary
fragmentation, not collective terminology and not a linking-vocabulary gap, and it is not
counted or characterised here. It is noted so it is not rediscovered as part of this defect; it
needs its own measurement before it is worth an entry.

⚠️ **What is NOT claimed.** No fix is proposed. Whether a class mention could be resolved to
the agent a paper is actually about is untested, and is a linking question rather than a
pairing one. **The yield gradient is measured and robust; the mechanism is only partly
resolved** — the sentence-boundary component is demonstrated by the table above, the
collective-terminology component is supported by the unlinked-surface evidence but not
quantified, and their relative weight is unknown.

---

## DEF-0004 — A link whose concept type contradicts the mention's own NER label is never refused, because the linker is never told the label

- **Date:** 2026-09-06
- **Component:** `biolit.canon` — the `Linker` protocol and `DictionaryLinker`
- **Status:** Recorded, **not fixed**. Opened as a stub before the DEF-0001 adjudication ran,
  because the tradeoff below is true regardless of how those labels came out; **decomposed
  2026-09-06** once they existed — 8 of the 10 flagged pairs are link errors, 1 is an NER span
  error and 1 is both.
- **Severity:** Wrong entity, silently, at full confidence — the same reader-visible damage as
  DEF-0001, but with a **mechanically detectable** signature that DEF-0001's class does not have.

### What was observed

`data/canon/concept_labels.json.gz` types each concept `CHEMICAL` or `DISEASE`. Comparing that
type against the NER label the mention already carries, across all eight frozen states:

| | |
|---|---|
| linked mentions | 3,696 |
| whose concept's own type contradicts the mention's NER label | **87 (2.4%)** |
| — mention labelled `CHEMICAL` → concept typed `DISEASE` | 59 |
| — mention labelled `DISEASE` → concept typed `CHEMICAL` | 28 |

Within DEF-0001's population of short all-caps pairs: **10 of 44 pairs, 74 of 204 mentions**,
including the largest pair by mention count.

```
26 mentions  APT  DISEASE  -> MESH:C071989 'APT'                              CHEMICAL
 9 mentions  GSH  CHEMICAL -> MESH:C563177 'Glucocorticoid-Remediable Aldo…'  DISEASE
 9 mentions  CP   CHEMICAL -> MESH:C566991 'Cleft Palate, Isolated, And Me…'  DISEASE
 8 mentions  CPA  CHEMICAL -> MESH:C537786 'Aphakia, congenital primary'      DISEASE
 8 mentions  RA   CHEMICAL -> MESH:D001172 'Arthritis, Rheumatoid'            DISEASE
 6 mentions  AT   CHEMICAL -> MESH:D001260 'Ataxia Telangiectasia'            DISEASE
 4 mentions  PCC  CHEMICAL -> OMIM:115700  'CATARACT 4, MULTIPLE TYPES'       DISEASE
 2 mentions  BLM  CHEMICAL -> MESH:D001816 'Bloom Syndrome'                   DISEASE
 1 mention   CD, 1 mention DIC  (pairs whose NER label varies across documents)
```

### Why it is possible

**`concept_labels.json.gz` has no consumer anywhere in `src/`.** It was built for ADR-0012's
embedding-fallback measurement, which was declined, and the artifact outlived the mechanism.

The deeper reason is structural rather than an unset flag: the `Linker` protocol is

```python
def link(self, surface: str) -> LinkResult: ...
```

It receives the surface and nothing else. **The mention's label is never passed to the linker**,
so this check is not switched off — it is unavailable on the shipped path. `EVAL_REPORT.md`
records that "label-constraining helps slightly and costs nothing: +0.0017 F1 (0.7880 →
0.7897)"; that was measured on a harness path that is not the one the pipeline runs.

### ⚠️ What a violation proves, and what it does not

**It proves an internal inconsistency, not a link error.** For `DIC` labelled `DISEASE` and
linked to `Dacarbazine` (typed `CHEMICAL`), either the text means disseminated intravascular
coagulation and the *link* is wrong, or it means the drug and the *NER label* is wrong. One of
the two is wrong; which one requires reading the abstract.

That distinction is why this entry claims 87 inconsistencies and **not** 87 wrong links.

### Why not fixed here

⛔ **The obvious fix is not free, and its cost lands on this project's dominant failure mode.**
Refusing a type-violating link converts a wrong entity into a NIL — and Phase 3 measured the
e2e NIL rate at **0.45 on the domain sample, 0.32 on BC5CDR**, recording the failure mode as
*abstention, not error*. Trading precision for recall in the direction the bottleneck already
runs is exactly the trade ADR-0011's rejection of dictionary enrichment turned on.

It also cannot be measured here: these eight states have no gold, so both sides of the trade
are unscoreable on them. The measurement belongs on BC5CDR, where a refused link that should
have been kept shows up as a loss instead of disappearing.

Per ADR-0013, the fix does not get built before that measurement exists. What this entry
establishes is that the signal is present, free, and currently discarded.

### ⭐ UPDATED 2026-09-06 — the decomposition, from the DEF-0001 labels

**All ten flagged pairs were labelled `wrong`.** None was `correct`, `granularity` or
`cant_tell`. Over the census the flag's precision is **10 / 10** and its recall over wrong pairs
is **10 / 23** — it catches under half the errors, and misses nothing it fires on.

⚠️ **DESCRIPTIVE ONLY, and the reason is not a formality.** The ten flagged pairs were shown to
the annotator as a table before labelling, so these labels cannot be the blind test of the flag
this was designed as (spec §7.1). This entry still rests on the mechanical inconsistency, which
needs no labels. What *is* new evidence is the decomposition below, because the annotator was
never told what any of these surfaces actually meant.

**Which side was wrong — the link, or the NER label?** §"What a violation proves" said a
violation identifies an inconsistency without saying which half caused it. The passages settle
it for 8 of 10:

| pair | what the passage says | at fault |
|---|---|---|
| `APT` | amiodarone-induced pulmonary toxicity | **link** |
| `GSH` | glutathione | **link** |
| `CP` | cisplatin | **link** |
| `CPA` | cyclophosphamide | **link** |
| `RA` | rosmarinic acid | **link** |
| `PCC` | prothrombin complex concentrate | **link** |
| `BLM` | bleomycin | **link** |
| `DIC` | disseminated intravascular coagulation | **link** |
| `AT` | *the letters `AT` taken out of `ATO` (atorvastatin)* | **NER span** |
| `CD` | a chemical probe name in one passage; conduct disorders in the other | **both** |

⛔ **`AT` is the case that stops the obvious fix from being obvious.** The mention is a
*fragment of a longer token* — NER cut `AT` out of `ATO` — so no linking decision could have
been right, and refusing the link would suppress a symptom while leaving a span defect
upstream. A type-constrained linker would score this as a success and fix nothing.

**What this does and does not change about the fix.** It strengthens the case that the signal is
real: 8 of 10 are squarely link errors that a type check would have caught for free. It does
**not** touch the reason the fix is not built here — refusing these links converts wrong
entities into NILs, and abstention is already this project's dominant failure mode. That trade
still needs BC5CDR, where both sides of it score.

**One number in this entry made more precise.** The "74 of 204 mentions" above counts
individually type-violating *mentions*; the ten flagged *pairs* carry **77** mentions in total.
The two differ because `CD` and `DIC` are labelled inconsistently across documents, so some of
their mentions violate and some do not. Both figures are correct about different things and the
distinction is now stated rather than left for a reader to trip over.

---

## DEF-0001 — Short acronym surfaces link confidently to whichever concept owns that acronym in CTD, with no context check

- **Date:** 2026-09-05
- **Component:** `biolit.canon` — `MeshDictionary.lookup` / `DictionaryLinker`
- **Status:** ⭐ **MEASURED 2026-09-06** — 23 of 44 pairs (52.3%) and 132 of 204 mentions (64.7%) carry a wrong concept. Still **not fixed**; no consumer-side workaround, see "Why not fixed here".
- **Severity:** Wrong entity, silently, at full confidence. `LinkResult.tiebroken` is `False`
  for these, so nothing downstream can tell them from a clean link.

### What was observed

`Amiodarone | Incontinentia Pigmenti` appears as a real 2-paper cluster in the frozen
`amiodarone pulmonary toxicity` run, and therefore in that query's answer. Verified in the
state dump:

```
surface 'IP'  label=DISEASE  ->  MESH:D007184 (Incontinentia Pigmenti)
```

In an amiodarone pulmonary-toxicity corpus `IP` is interstitial pneumonitis. It linked to a
dermatological genetic disorder because that concept owns the alias `IP` in the CTD table,
the alias is unambiguous *in the table*, and nothing consults the surrounding text.

The same run's largest cluster has the same shape: `APT` → `MESH:C071989`, where in context
APT is the query's own subject, amiodarone pulmonary toxicity.

### The class, measured

Across all eight frozen states (5,969 entity mentions):

| | |
|---|---|
| linked mentions whose surface is all-caps and ≤ 4 characters | **204 (3.4%)** |
| distinct (surface, concept) pairs among them | **44** |

⚠️ **Not all 204 are errors, and the entry does not claim they are.** Several are correct —
`ICH` → Cerebral Hemorrhage, `ATP` → Adenosine Triphosphate, `HCC` → Carcinoma Hepatocellular,
`FXS` → Fragile X Syndrome. That is precisely the defect: correctness here is decided by which
concept happens to own the acronym in CTD, not by anything about the document. Verified
examples from the wrong side of that coin:

| surface | linked to | what the abstract itself says |
|---|---|---|
| `GSH` | Glucocorticoid-Remediable Aldosteronism | `…Malondialdehyde (MDA) and glutathione (GSH)` |
| `ATN` | Oculocutaneous albinism type 1 | `…prior platinum-associated acute tubular necrosis (ATN)` |
| `CP` | Cleft Palate, Isolated, And Mental Retardation | `…Apoptosis In Vivo and In Vitro. Cisplatin (CP)` |
| `CPA` | Aphakia, congenital primary | `…methotrexate (MTX), cyclophosphamide (CPA)` |
| `AITC` | 2,3,4-tri-O-acetylarabinopyranosyl isothiocyanate | `…(HEK)-293 cells. Allyl isothiocyanate (AITC)` |

⭐ **AMENDED 2026-09-06 — the third column was originally this entry's own guesses, headed "in
context almost certainly", and two of the five were wrong.** `CP` was guessed as creatine
phosphokinase / cardiopulmonary and is **cisplatin**; `CPA` was guessed as cyproterone acetate
and is **cyclophosphamide**. The column now quotes the source abstracts verbatim instead.

**This is the entry's own caution being vindicated in a specific way, and it is kept rather
than tidied:** the entry refused to state a rate because the pairs had not been adjudicated,
and the informal reading it did offer — on the five rows it was most confident about — was
wrong on two. A rate derived from that reading would have been wrong in a way nothing would
have caught.

### ⭐ MEASURED 2026-09-06 — the rate, at last

All 44 pairs were adjudicated blind against their source passages, plus 8 controls, under gates
fixed before any label existed (`docs/superpowers/specs/2026-09-06-acronym-adjudication-design.md`;
`rows_hash ad22f212d9a39b55`, `labels_hash 538e8b522f04b370`).

| | per pair | per mention |
|---|---|---|
| **`wrong`** — the concept is not what the author meant | **23 / 44 (52.3%)** | **132 / 204 (64.7%)** |
| `granularity` — right subject, wrong level (DEF-0002's shape) | 2 / 44 | 18 / 204 |
| `correct` | 19 / 44 | 54 / 204 |
| `cant_tell` | **0** | — |

**A majority of these links are wrong, and the reader-facing figure is worse than the
mechanism-facing one.** Per pair asks how often the mechanism errs; per mention asks how much
wrong text a reader sees. The gap is not noise — the wrong pairs are the frequent ones, led by
`APT` at 26 mentions.

⚠️ **This is a CENSUS of these eight queries, not an estimate.** The 44 pairs are the complete
enumeration of the class in this corpus, so there is no sampling distribution and **no interval
is reported**. Whether 52% transfers to other queries is a question this design cannot answer.

**Reported split, per ADR-0016 rule 6**, because 17 of the 44 were named to the annotator before
labelling:

| | pairs `wrong` | mentions `wrong` |
|---|---|---|
| disclosed (17) | 13 / 17 | 93 / 131 |
| **undisclosed (27)** | **10 / 27** | **39 / 73** |

⛔ **The gap between the two rows must NOT be read as disclosure bias.** The disclosed set was
*selected for looking wrong* — `DEFECTS.md` picked known-bad examples and DEF-0004's flag picked
inconsistencies — so a higher rate there is expected by construction, disclosure or not.
Selection and disclosure are confounded and this design cannot separate them. What the split
does establish is that **on 27 pairs carrying no prior disclosure at all, 10 are still wrong**.

**Two signs the labels are independent judgment rather than an echo.** The annotator marked
`ICH` as `granularity` where this entry had named it *correct* — reading intracranial
haemorrhage as broader than `Cerebral Hemorrhage`, which is DEF-0002's exact shape. And for the
six pairs disclosed only as type-violating, with no direction attached (`RA`, `AT`, `PCC`,
`BLM`, `CD`, `DIC`), the recorded reasons name meanings that were never disclosed to them —
rosmarinic acid, prothrombin complex concentrate, bleomycin, conduct disorders.

**A caveat on Gate 2 belongs with this reading rather than buried in the spec:** the controls
were structurally identifiable by surface duplication — every control kept its surface, and over
a census that surface is already in the sheet as a real row. The label pattern refutes the
shortcut having been used (four duplicated surfaces had *both* members marked `wrong`, which a
duplicate-spotter cannot produce), but the instrument was not clean. Generalised as
**ADR-0021**, which is a design-methodology finding rather than an entity-linking one and so
lives in `DECISIONS.md` rather than here.

See also **DEF-0004**, which the adjudication decomposes.

### Why it is not a selection or synthesis defect

It reaches the reader through the answer, which is where it was noticed, but nothing in
`select_stage` or `synthesis_stage` could have prevented it. By the time a cluster exists,
`IP` *is* Incontinentia Pigmenti as far as every downstream stage can tell: the concept id is
well-formed, the cluster key is legitimate, two papers really do share it, and `select_stage`
correctly keeps it because Amiodarone is a query concept. Filtering it out downstream would
mean suppressing a correctly-formed cluster on a guess, which is worse than showing it.

### Why not fixed here

A fix belongs in linking and needs a design, not a patch. The obvious candidates each carry a
cost that has to be measured rather than assumed:

- **Refuse short all-caps surfaces outright.** Cheap, and it would discard `ICH`, `ATP`,
  `HCC`, `FXS` along with the errors — trading a precision problem for a recall problem in a
  pipeline whose measured bottleneck is already abstention (Phase 3 recorded the failure mode
  as *abstention, not error*).
- **Require the acronym's expansion to appear in the same document.** The standard approach,
  and the right shape, since abstracts usually define an acronym on first use. Real work: a
  definition-detection pass plus a decision about what to do when no definition is found.
- **Score candidates against document context.** ADR-0012 already measured and declined the
  embedding fallback linker as too weak and too costly for its benefit; that reading was about
  a different problem, but it is the nearest prior and should be consulted before anyone
  proposes embeddings here.

Per ADR-0013, none of that gets built before someone decides which failure this project would
rather have.

---

## DEF-0002 — A bare parent-concept mention is indistinguishable from its specific child

- **Date:** 2026-09-04 (recorded in ADR-0019; moved here 2026-09-05 so it is findable)
- **Component:** `biolit.canon` — same context-free lookup as DEF-0001
- **Status:** **Recorded, not fixed — but PARTIALLY ROUTED AROUND downstream since 2026-09-07.**
  The linking defect below is untouched. What changed is that `select_stage` no longer *drops*
  clusters over one instance of this granularity mismatch on the chemical side (ADR-0022); the
  disease side and the linker itself are unaffected. See the update at the end of this entry —
  the status line is qualified rather than left reading clean, because a reader scanning
  statuses would otherwise miss that half the observed damage is now handled elsewhere.

A bare `TNF` resolves to the parent MeSH concept of a source's `TNF-α`, so concept
**hierarchy** produces a false entity-hallucination positive by a different mechanism than
the Unicode-normalisation class fixed on the Gate A branch. It was left unfixed there because
it was moot for a retired gate — see ADR-0019's consequences section, which carries the full
context.

It is logged here because it is not actually specific to that gate: it is the same root cause
as DEF-0001 — a surface form is linked without reference to what the document is about — and
any fix for one should be checked against the other.

**⭐ UPDATE 2026-09-07 — the SELECTION half of this shape is closed on the chemical side; the
LINKING half is untouched.** Gate 3 of the relevance pass found the same granularity mismatch
arriving at a different stage: `select_stage` deleted three `answers` clusters carrying
`Atorvastatin` because the query resolved to the drug **class**, class-to-member on the
chemical side. **ADR-0022 fixes that** — a cluster side now matches if it belongs to a
pharmacological class the query named, and all three are recovered. ⚠️ **This is not a fix for
the defect above, and the distinction is the point.** ADR-0022 changes what the *filter* does
with two correctly-linked concepts whose granularity differs; DEF-0002 is about the *linker*
producing the wrong concept from an ambiguous surface, and nothing here gives the linker any
context it did not have. The disease side is also still open: no equivalent relation was built
for diseases, and MeSH's `PharmacologicalAction` field only covers chemicals. **What ADR-0022
does establish for any future attempt here: the MeSH tree cannot express a drug class's
membership at all** — `Atorvastatin` is filed under chemical structure (D03/D10) and its class
under actions and uses (D27), sharing no node — so a hierarchy walk is the wrong instrument for
this half of the problem, whichever stage it is attempted at.

---

## DEF-0003 — ADR-0020's hierarchy term is unreachable: every cluster the ranker sees scores proximity 0

- **Date:** 2026-09-05
- **Component:** `biolit.query.ranking._relevance_key`
- **Status:** ⭐ **FIXED 2026-09-05 as a structural revision — and UNVALIDATED.** Proximity is
  now the maximum over sides that are not already matched. ⚠️ **Wording updated 2026-09-08:
  this said "not already *exact* matches", which ADR-0022 made imprecise** — a side now counts
  as matched if it IS a query concept **or** belongs to a pharmacological class the query named,
  and the residual exclusion widened with it. That was deliberate rather than incidental: a
  class member shares no tree node with its class, so a matched-but-unexcluded side would score
  `inf` and sort last — this defect's masking shape arriving through a new door. The invariant
  is preserved rather than merely still true. Verified mechanically only:
  queries whose clusters all tie fell from 4 of 8 to 2 of 8. **These labels are spent as a
  blind test of the fix** and were not re-read; see ADR-0020's addendum. ⚠️ **Added 2026-09-06:**
  they also come from a pass whose control instrument was later found compromised (ADR-0021), so
  they could not have validated the fix even if spending them were permitted. **This fix rests
  on the mechanical check and on nothing else** — not on a null result from the labels, which is
  a different and weaker claim.
- **Found by:** Gate 4 of the cluster-relevance annotation pass, which is what it was for.

### What was observed

ADR-0020 scores a cluster as `(-exact_matches, min_tree_distance, key)`. The second term
never varies. Measured over the frozen corpus, **four of eight queries produce a single
distinct score across every cluster** — amiodarone, cisplatin, isotretinoin, lithium — so the
ranker falls back entirely to the MeSH-id order it was built to replace.

The mechanism, verified on `isotretinoin and depression`:

```
query concepts: [MESH:D003866 Depressive Disorder, MESH:D015474 Isotretinoin]
  Acne Vulgaris              score=(-1, 0)   disease-side distance to D003866: None
  Anxiety Disorders          score=(-1, 0)   disease-side distance to D003866: 3
  Mental Disorders           score=(-1, 0)   disease-side distance to D003866: 2
  Psychotic Disorders        score=(-1, 0)   disease-side distance to D003866: 4
```

The distances ADR-0020 relies on are computed correctly and then discarded. `_relevance_key`
minimises over **every** (cluster side × query concept) pair, and the cluster's own chemical
side *is* a query concept — `distance(Isotretinoin, Isotretinoin) == 0` — so the minimum is 0
for every cluster and the disease-side signal never reaches the sort.

⛔ **This is total, not partial.** `select_stage` keeps only clusters with at least one exact
concept match, and any exact match contributes distance 0. **Every cluster the ranker is ever
handed therefore scores proximity 0**, so the term is unreachable in production by
construction. The score is operationally `(-exact_matches, key)`.

### What that means for the reading it produced

Gate 4a improved leads from 3/8 to 5/8, and **that improvement came entirely from the
exact-match count**. Nothing in it is evidence about hierarchy proximity, which is the part of
ADR-0020 the MeSH tree artifact exists to serve. The distances quoted in ADR-0020's decision
section — Mental Disorders 2, Anxiety 3, Psychotic 4, Acne no shared tree — are correct
readings of the tree and were never reachable by the code.

### Why this is not simply fixed

The fix is structural rather than a constant: proximity should describe the **residual** match
quality of sides that are not already exact — a cluster is as good as its *worst*-matched
side, so the term wants a max (or a sum) over sides rather than a min over all pairs. Under
that reading, `Isotretinoin | Anxiety Disorders` scores 3 and `Isotretinoin | Acne Vulgaris`
scores unmatched, which is the ordering ADR-0020 intended.

⚠️ **The labels this defect was found with are spent as a blind test of the fix.** §5 permits a
revision changing the score's structure and forbids one tuning a constant, and max-over-sides is
squarely the permitted kind — but it would be measured against labels chosen *before* the defect
was known and read *after*. **No Gate 4 re-run was performed**, and none should be quoted as
validating this. Evidence about hierarchy proximity needs a fresh label set or queries outside
these eight.

**What the fix does not fix.** Two queries still tie across every cluster: `amiodarone pulmonary
toxicity` and `cisplatin nephrotoxicity`. Their disease terms resolve to no MeSH concept even
through NCBI, so the query is a lone chemical and every kept cluster carries it, while the
disease sides share no tree with any chemical. That is a query-linking limit, not a scoring one,
and it is recorded in ADR-0020's addendum rather than here.
