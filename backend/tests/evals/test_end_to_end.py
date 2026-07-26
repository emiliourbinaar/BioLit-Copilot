import json

import pytest

from biolit.canon.linker import DictionaryLinker
from biolit.canon.mesh import AliasEntry, MeshConcept, MeshDictionary
from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity
from biolit_evals.end_to_end import concept_counts, metrics_from_counts, score_end_to_end
from biolit_evals.mesh_gold import GoldDocument, GoldMention

CHEMICAL = EntityLabel.CHEMICAL


def _linker():
    met = MeshConcept(id="MESH:D008687", name="Metformin")
    return DictionaryLinker(MeshDictionary({"metformin": [AliasEntry(met, True)]}))


def test_score_end_to_end_scores_concepts_and_censuses_outcomes():
    doc = GoldDocument(
        pmid="1",
        text="metformin treats PCOS",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="metformin",
                label=CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=17,
                end=21,
                text="PCOS",
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D011085",),
            ),
        ],
    )
    # Predict the chemical exactly; miss the disease entirely.
    preds = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=_linker())

    assert m.n_documents == 1
    assert (m.concepts.tp, m.concepts.fp, m.concepts.fn) == (1, 0, 1)
    assert m.census.outcomes == {"EXACT": 1, "MISSED": 1}
    assert m.concepts_by_label["CHEMICAL"].tp == 1
    assert m.concepts_by_label["DISEASE"].fn == 1
    assert m.n_predicted == 1 and m.n_predicted_linked == 1
    assert m.e2e_nil_rate == pytest.approx(0.0)
    assert m.merge_candidates == 0


def test_concept_counts_for_one_document():
    tp, fp, fn = concept_counts({"MESH:A", "MESH:B"}, {"MESH:A", "MESH:C"})
    assert (tp, fp, fn) == (1, 1, 1)


def test_metrics_are_micro_averaged_not_macro():
    # Doc 1: tp=1 fp=0 fn=0 (perfect). Doc 2: tp=1 fp=3 fn=0 (precision 0.25).
    # Micro precision over pooled totals = 2/5 = 0.4, while the MACRO average of the
    # two per-document precisions would be (1.0 + 0.25)/2 = 0.625. Pinning the micro value
    # is what makes a macro-average regression fail this test.
    d1 = concept_counts({"MESH:A"}, {"MESH:A"})
    d2 = concept_counts({"MESH:B"}, {"MESH:B", "MESH:X", "MESH:Y", "MESH:Z"})
    tp = d1[0] + d2[0]
    fp = d1[1] + d2[1]
    fn = d1[2] + d2[2]
    m = metrics_from_counts(tp, fp, fn)
    assert (m.tp, m.fp, m.fn) == (2, 3, 0)
    assert m.precision == pytest.approx(2 / 5)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(2 * (2 / 5) * 1.0 / ((2 / 5) + 1.0))


def test_metrics_from_counts_all_zero_is_safe():
    m = metrics_from_counts(0, 0, 0)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_repeated_concept_in_one_document_counts_once():
    # Both gold and predictions mention the same concept repeatedly; sets collapse it, so
    # a single much-repeated entity cannot dominate the corpus totals.
    tp, fp, fn = concept_counts({"MESH:A"}, {"MESH:A"})
    assert (tp, fp, fn) == (1, 0, 0)


def test_merge_audit_counts_candidates_and_nil_gap_fills():
    # "GLP" + "1RA" is the ADR-0008 shape: neither fragment links alone, the merged
    # surface does, so both constituents inherit it and the audit records one candidate.
    glp = MeshConcept(id="MESH:D000067299", name="GLP-1 Receptor Agonists")
    linker = DictionaryLinker(MeshDictionary({"glp-1ra": [AliasEntry(glp, True)]}))
    doc = GoldDocument(
        pmid="1",
        text="GLP-1RA therapy",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=7,
                text="GLP-1RA",
                label=CHEMICAL,
                mesh_ids=("MESH:D000067299",),
            )
        ],
    )
    preds = [
        Entity(text="GLP", label=CHEMICAL, start=0, end=3),
        Entity(text="1RA", label=CHEMICAL, start=4, end=7),
    ]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=linker)
    assert m.merge_candidates == 1
    assert m.merge_candidates_linked == 1
    assert m.merge_candidates_matching_gold == 1
    assert m.merged_constituents == 2
    assert m.census.outcomes == {"MERGEABLE": 1}
    assert m.e2e_nil_rate == pytest.approx(0.0)


def test_exact_link_audit_partitions_the_census_exact_cell():
    # Three gold mentions: one exact span the dictionary knows, one exact span it does not
    # (the fallback-addressable case), one never predicted at all. The audit must cover the
    # first two and only those -- its total is exactly the census EXACT count.
    doc = GoldDocument(
        pmid="1",
        text="metformin and canagliflozin treat PCOS",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="metformin",
                label=CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=14,
                end=27,
                text="canagliflozin",
                label=CHEMICAL,
                mesh_ids=("MESH:C000589404",),
            ),
            GoldMention(
                pmid="1",
                start=34,
                end=38,
                text="PCOS",
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D011085",),
            ),
        ],
    )
    preds = [
        Entity(text="metformin", label=CHEMICAL, start=0, end=9),
        Entity(text="canagliflozin", label=CHEMICAL, start=14, end=27),
    ]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=_linker())

    assert m.census.outcomes == {"EXACT": 2, "MISSED": 1}
    assert m.exact_link.total == m.census.outcomes["EXACT"]
    assert m.exact_link.statuses == {"LINKED_CORRECT": 1, "NIL": 1}
    assert m.exact_link.by_label == {"CHEMICAL": {"LINKED_CORRECT": 1, "NIL": 1}}


def test_oracle_ceiling_grants_gold_ids_to_exact_spans_that_went_nil():
    # "canagliflozin" is spanned exactly but absent from the dictionary -- the addressable
    # case. The oracle grants it its gold id, converting a concept-level fn into a tp; the
    # baseline metrics alongside it must be untouched.
    doc = GoldDocument(
        pmid="1",
        text="metformin and canagliflozin",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="metformin",
                label=CHEMICAL,
                mesh_ids=("MESH:D008687",),
            ),
            GoldMention(
                pmid="1",
                start=14,
                end=27,
                text="canagliflozin",
                label=CHEMICAL,
                mesh_ids=("MESH:C000589404",),
            ),
        ],
    )
    preds = [
        Entity(text="metformin", label=CHEMICAL, start=0, end=9),
        Entity(text="canagliflozin", label=CHEMICAL, start=14, end=27),
    ]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=_linker())

    assert (m.concepts.tp, m.concepts.fn) == (1, 1)
    assert m.oracle.n_granted == 1
    assert (m.oracle.concepts.tp, m.oracle.concepts.fn) == (2, 0)
    assert m.oracle.concepts_by_label["CHEMICAL"].tp == 2


def test_oracle_grants_nothing_for_wrong_links_or_damaged_spans():
    # This is what keeps the ceiling a *fallback* bound rather than a general linking bound.
    # Mention 1: exact span, dictionary answers confidently and wrongly -- a fallback is
    # never consulted, so no grant. Mention 2: truncated span ("metf" for "metformin") --
    # not an exact span, so its linking outcome says nothing about dictionary coverage.
    doc = GoldDocument(
        pmid="1",
        text="metformin and metformin",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="metformin",
                label=CHEMICAL,
                mesh_ids=("MESH:D999",),  # dictionary will return MESH:D008687 -> WRONG
            ),
            GoldMention(
                pmid="1",
                start=14,
                end=23,
                text="metformin",
                label=CHEMICAL,
                mesh_ids=("MESH:D777",),
            ),
        ],
    )
    preds = [
        Entity(text="metformin", label=CHEMICAL, start=0, end=9),
        Entity(text="metf", label=CHEMICAL, start=14, end=18),
    ]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=_linker())

    assert m.exact_link.statuses == {"LINKED_WRONG": 1}
    assert m.census.outcomes == {"EXACT": 1, "TRUNCATED": 1}
    assert m.oracle.n_granted == 0
    assert m.oracle.concepts == m.concepts  # ceiling collapses onto the baseline


def test_oracle_splits_into_abbreviation_addressable_and_the_rest():
    # "TdP" is defined in the document and its long form links, so deterministic in-document
    # expansion would reach it. "hepatic injury" is a paraphrase the dictionary cannot reach
    # by any expansion. The split must attribute one to each.
    tdp = MeshConcept(id="MESH:D016171", name="Torsades de Pointes")
    linker = DictionaryLinker(MeshDictionary({"torsades de pointes": [AliasEntry(tdp, True)]}))
    # Document-length text, so this exercises the real path rather than the fragment guard.
    text = (
        "Torsades de pointes (TdP) is a recognised complication of QT-prolonging therapy. "
        "We reviewed two hundred and forty consecutive admissions across a four-year period "
        "at a single tertiary centre and assessed the outcomes. TdP and hepatic injury were "
        "seen in a minority of patients, and all episodes resolved after withdrawal."
    )
    tdp_at = text.index("TdP", text.index("assessed"))
    inj_at = text.index("hepatic injury")
    doc = GoldDocument(
        pmid="1",
        text=text,
        mentions=[
            GoldMention(
                pmid="1",
                start=tdp_at,
                end=tdp_at + 3,
                text="TdP",
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D016171",),
            ),
            GoldMention(
                pmid="1",
                start=inj_at,
                end=inj_at + 14,
                text="hepatic injury",
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D056486",),
            ),
        ],
    )
    preds = [
        Entity(text="TdP", label=EntityLabel.DISEASE, start=tdp_at, end=tdp_at + 3),
        Entity(text="hepatic injury", label=EntityLabel.DISEASE, start=inj_at, end=inj_at + 14),
    ]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=linker)

    assert m.oracle.n_granted == 2
    assert m.oracle_abbrev.n_granted == 1
    assert m.oracle_paraphrase.n_granted == 1
    # The abbreviation slice recovers the concept its expansion resolves to, and only that.
    assert m.oracle_abbrev.concepts.tp == 1
    assert m.oracle_paraphrase.concepts.tp == 1


def test_fragment_documents_skip_abbreviation_expansion_and_are_counted():
    # A single sentence cannot contain a definition that licenses expansion elsewhere in
    # the document. Rather than silently finding nothing -- indistinguishable from "no
    # abbreviations here" -- the run records that it declined to try.
    tdp = MeshConcept(id="MESH:D016171", name="Torsades de Pointes")
    linker = DictionaryLinker(MeshDictionary({"torsades de pointes": [AliasEntry(tdp, True)]}))
    doc = GoldDocument(
        pmid="1",
        text="TdP occurred.",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=3,
                text="TdP",
                label=EntityLabel.DISEASE,
                mesh_ids=("MESH:D016171",),
            )
        ],
    )
    preds = [Entity(text="TdP", label=EntityLabel.DISEASE, start=0, end=3)]
    m = score_end_to_end([doc], predict=lambda _t: preds, linker=linker)

    assert m.abbrev_context_skipped == 1
    # The mention is still counted in the ceiling -- it just cannot be credited to the
    # deterministic slice on evidence this document does not contain.
    assert m.oracle.n_granted == 1
    assert m.oracle_abbrev.n_granted == 0


_EXPECTED_KEYS = {
    "timestamp",
    "git_sha",
    "dataset",
    "artifact_source",
    "n_aliases",
    "n_documents",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "e2e_nil_rate",
    "n_predicted",
    "n_predicted_linked",
    "census",
    "concepts_by_label",
    "exact_link",
    "oracle_exact_nil",
    "abbrev_context_skipped",
    "oracle_abbrev",
    "oracle_paraphrase",
    "merge_audit",
}


def test_run_e2e_eval_appends_one_log_line_with_exact_schema(tmp_path):
    from biolit_evals.end_to_end import run_e2e_eval

    doc = GoldDocument(
        pmid="1",
        text="metformin",
        mentions=[
            GoldMention(
                pmid="1",
                start=0,
                end=9,
                text="metformin",
                label=CHEMICAL,
                mesh_ids=("MESH:D008687",),
            )
        ],
    )
    preds = [Entity(text="metformin", label=CHEMICAL, start=0, end=9)]
    log = tmp_path / "nested" / "e2e_runs.jsonl"
    m = run_e2e_eval(
        documents=[doc],
        predict=lambda _t: preds,
        linker=_linker(),
        dataset="fixture",
        artifact_source="fixture",
        n_aliases=1,
        log_path=str(log),
        git_sha="abc1234",
        now="2026-07-23T00:00:00+00:00",
    )
    assert m.concepts.tp == 1
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record.keys()) == _EXPECTED_KEYS
    assert record["census"]["outcomes"] == {"EXACT": 1}
