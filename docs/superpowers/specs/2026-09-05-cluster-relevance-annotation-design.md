# Cluster-relevance annotation — validating query-conditioned selection on the frozen corpus

- **Date:** 2026-09-05
- **Status:** **Design. Not yet run.** No labels exist. Nothing in the pipeline depends on this
  reading, and `select_stage` ships without it — this validates a filter already in production
  rather than gating its release.
- **Kind:** Measurement. A validity probe on a shipped deterministic component, run against a
  corpus that was frozen for an unrelated purpose and therefore cannot have been selected to
  flatter it.
- **Depends on:** ADR-0017 (π is a **ceiling**, never a correction factor; readability/preference
  is a forbidden metric), ADR-0018 (a single annotator needs a discriminating control before any
  reading is attributable), ADR-0016 (verification rules; decide by running the weakened case),
  ADR-0013 (no infrastructure without a demonstrated consumer), ADR-0015 (an arm scored on a
  population it cannot lose on proves nothing).
- **Does not decide:** whether clusters should be **ordered** by relevance. §7 explains why the
  labels will nonetheless be the evidence a future ordering decision needs, and why collecting
  them costs nothing extra.

---

## 1. What this measures, and why it is worth 83 labels

`select_stage` keeps a cluster when either side of its `chemical|disease` key is a MeSH concept
NCBI translated the query into. That is a deterministic, checkable rule, and on the frozen
corpus it keeps **70 of 83 clusters**. What no measurement yet says is whether those are the
*right* 70.

The question is small enough to answer exactly. The corpus is already frozen — eight PubMed
queries, 470 papers, 83 clusters after Ruling 26's merge-by-key — and every cluster is a short,
inspectable object: a chemical concept, a disease concept, and a list of papers. **83 labels
covers the entire population, not a sample.** There is no sampling error to reason about and no
stratification to design, which is the opposite of every prior annotation pass in this project.

⚠️ **This is a sanity check on a filter, not a gold standard.** One annotator produces it. It
cannot support an inter-annotator agreement figure, it must never be used to tune a threshold
into existence, and it must never be used to score a generative arm. Its only sanctioned uses
are §6's.

---

## 2. Unit and label schema

**Unit:** one `(query, cluster)` pair. 83 real rows.

The annotator sees the query, the cluster's two concept names, the paper count, and the year
range. **Not** the cluster key's raw MeSH ids, and **not** the finding sentences — the judgment
is about topical relevance, and reading the findings invites judging quality instead.

| label | meaning |
|---|---|
| `answers` | This cluster is part of what the question asked for. |
| `background` | Genuinely about the query's subject, but not what was asked — the drug's indication, its main comparator, a co-occurring condition. |
| `off_topic` | Not what the question was about. |
| `cant_tell` | The query, the concepts, or the pairing is too ambiguous to judge. |

**Why `background` is a separate class and not folded into either neighbour.** It is the class
the whole design turns on. `Isotretinoin | Acne Vulgaris` (18 papers) is not off-topic for
"isotretinoin and depression" — it is the indication, and a reader would reasonably want it —
but it is not what was asked, and it is currently what the answer *opens with*. Collapsing it
into `answers` would declare the current behaviour correct by definition; collapsing it into
`off_topic` would demand the filter drop it, which is probably wrong. Keeping it separate is
what lets the same 83 labels speak to both the filter and to ordering (§7).

`cant_tell` exists as a **tractability gate**, exactly as in ADR-0018's Gate 1. It is not a
convenience escape hatch, and §5 makes a high rate fatal to the whole reading rather than
something to work around.

---

## 3. Protocol

1. **Write the label definitions first.** §2 is fixed before any row is seen. It is not revised
   mid-pass; if it proves inadequate, the pass stops and restarts with a new definition and no
   labels carried over.
2. **Blind to the filter.** The export **must not** carry `select_stage`'s keep/drop decision,
   and must not be ordered by it. Rows are shuffled under a recorded seed across all queries, so
   consecutive rows come from different questions. Without this the labels measure agreement
   with a decision already seen, which is not a measurement.
3. **Blind to cluster size ordering.** Rows are not sorted by paper count. Size correlates with
   relevance and would leak an ordering cue.
4. **Distractor controls, per ADR-0018.** Inject **8 distractor rows**: a cluster drawn from one
   query, presented under a *different* query, chosen so the pairing is genuinely unrelated
   (e.g. `Warfarin | Thromboembolism` presented under "cisplatin nephrotoxicity"). They are
   shuffled in with the real rows and are indistinguishable in the export. **91 rows total.**
5. **One pass, no revisiting.** Labels are appended as they are made; earlier rows are not
   revised after later ones clarify the schema. Revisiting silently converts a blind pass into a
   calibrated one.
6. **The annotator is the project owner.** This is stated as a limitation, not managed away —
   see §8.

---

## 4. Export format

One JSONL file, one row per line, written to `data/relevance/` (gitignored, like every other
corpus artifact) with a manifest hash committed to the run log alongside the seed.

```json
{"row_id": "r07", "query": "isotretinoin and depression",
 "chemical": "Isotretinoin", "disease": "Acne Vulgaris", "n_papers": 18,
 "year_range": "1990-2024", "label": null}
```

`row_id` is opaque and carries no information about query, source cluster, or distractor status.
The mapping from `row_id` back to `(query, cluster_key, is_distractor, filter_kept)` lives in a
**separate** manifest file that the annotator does not open until every label is written.

Content hashing follows `synth_corpus.sample_hash`: sorted, content-addressed, stable over
order and serialization, so the exact row set is pinned before any label exists.

---

## 5. Gates and pre-registered readings

Written before any label exists. **Numbers are read as they fall; no threshold moves to match
an observation.**

**Gate 1 — tractability.** `cant_tell` over the 83 real rows.
- ≥ 15% → **`REVISE_SCHEMA`**: the label definitions do not fit the data. **Stop. Do not compute
  Gate 2, and do not report the filter's numbers at all** — a filter reading computed over rows
  the annotator could not judge is not interpretable. The precedent is ADR-0018's Gate 1, which
  was a genuine precondition and not a formality.
- < 15% → proceed.

**Gate 2 — control discrimination.** The 8 distractors.
- ≥ 7/8 labelled `off_topic` → **`DISCRIMINATING`**. The annotator is using the label for its
  meaning, and the Gate 3 reading is attributable to the filter.
- < 7/8 → **`CONFOUNDED`**. Report Gate 3 descriptively and draw no conclusion from it. This is
  ADR-0018's contribution: without this control, a permissive annotator and a good filter are
  indistinguishable.

**Gate 3 — the filter's confusion matrix**, over the 83 real rows, with 95% Wilson intervals on
each rate and the raw counts always printed beside them.

|  | labelled `answers` | labelled `background` | labelled `off_topic` |
|---|---|---|---|
| **kept** (70) | correct | *the ordering question* | over-inclusion |
| **dropped** (13) | ⛔ **false drop** | under-inclusion | correct |

**Pre-registered actions, so the reading cannot be rationalised afterwards:**

- ⛔ **Any `answers` cluster the filter dropped is a defect to fix, not a rate to tolerate.**
  The population is 83 and each case is individually inspectable, so "an acceptable false-drop
  rate" is not a meaningful object here. Each one gets diagnosed to its cause — a query term
  that failed to link, a concept-granularity mismatch, an OR that should have matched — and
  fixed or explicitly recorded as unfixable.
- **`background` clusters that were kept are the ordering evidence (§7), not filter errors.**
  The filter is not asked to drop them and is not scored down for keeping them.
- **`off_topic` clusters that were kept measure over-inclusion**, and are reported as a rate
  with its interval. No threshold is pre-committed, because there is no principled one to
  commit to — the number is reported and read, and if it is bad the response is a diagnosed
  change to the rule, never a tuned constant.

**Power is stated up front, not discovered afterwards.** 13 dropped clusters is a small
denominator: one unexpected false drop moves that rate by roughly 8 points. Any statement about
the *drop* side is therefore weak evidence about a rate and strong evidence about a specific
case — which is precisely why the pre-registered action above is case-by-case rather than
rate-based. ADR-0018's underpowering was a defect because the design leaned on a rate it could
not resolve; this design deliberately does not lean on one.

---

## 6. Sanctioned uses of the labels

1. Diagnosing and fixing individual false drops (§5).
2. Reporting the filter's over-inclusion rate with its interval.
3. Deciding whether the ordering question is live (§7).
4. Validating a derived proxy on its own terms (§8.2) — and *only* by comparison against these
   labels, never by assuming the proxy.

**Explicitly not sanctioned:** tuning any constant in `select_stage`; scoring any generative
output; training anything; reporting an accuracy figure detached from its Gate 1 and Gate 2
verdicts.

---

## 7. What these labels give the deferred ordering decision

Ordering was deliberately not built. `render_cluster` states that its ordering "carries no
implicit ranking — a reader must not be able to infer importance from position", and overturning
that is an ADR, not a patch.

The **measured** case for reopening it is already strong and does not depend on this pass:
selection keeps 70 of 83 clusters and does not change what an answer leads with. "isotretinoin
and depression" keeps 5 of 5 and still opens on `Isotretinoin | Acne Vulgaris`.

What these labels add is the missing quantity: **how many kept clusters are `background` rather
than `answers`, and how often a `background` cluster currently precedes an `answers` one under
MeSH-id ordering.** That is computable from the labels with no extra annotation, it is the exact
input an ordering ADR needs, and collecting it costs nothing beyond keeping `background` as a
distinct class in §2. If the count is small the ordering question closes cheaply; if it is large
the ADR has its evidence.

---

## 8. Limitations, stated rather than managed

### 8.1 Single non-independent annotator

The annotator built the system. Blinding to the filter's decision (§3.2), shuffling (§3.2),
withholding the finding sentences (§2), and the distractor control (§3.4) are real mitigations
and they are not a substitute for independence. **No inter-annotator agreement figure can be
computed from this pass, and none should be quoted.** The honest description is: a structured,
blinded, single-annotator sanity check over a complete population, with a control that says
whether the labels track their definitions.

### 8.2 The MeSH-indexing cross-check is a candidate, not a shortcut

PubMed indexes every paper with MeSH descriptors, which suggests a free derived relevance
proxy: score a cluster by the overlap between its concepts and the descriptors on its own
papers. **It is not trusted, and it is not used to produce labels.**

This project has already spent one full phase discovering that a plausible derived proxy meant
something other than what it appeared to mean — the CTD contradiction proxy read π̂ = 0.067 with
an upper bound of 0.298, below anything usable (ADR-0017). The standing rule that came out of
that is: **validate a derived label's meaning before spending against it.**

So the ordering is fixed and non-negotiable: **human labels first, then the proxy is measured
against them.** Agreement is reported as a ceiling, per ADR-0017 — never as a correction factor,
and never used to "extend" the human labels to unlabelled data unless a separate decision
authorises it on the strength of the measured agreement. If the proxy disagrees with the human
labels, the proxy is discarded, not the labels.

### 8.3 Eight queries, all drug–adverse-effect shaped

The frozen corpus is eight queries of one clinical shape. Findings generalise to that shape and
are not evidence about mechanism questions, comparative-effectiveness questions, or anything
outside chemical–disease pairing. Stated because the corpus was frozen for Gate A and inherited
here, which is a genuine strength for freedom-from-selection-bias and a genuine limit on
external validity.
