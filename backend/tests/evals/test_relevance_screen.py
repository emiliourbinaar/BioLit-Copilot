import json
import subprocess
import sys

from biolit_evals.relevance_screen import ScreenRow, screen_hash, select

# ⚠️ The modules that can compute, or lead to computing, a difference between the arms.
# `stages` is on the list because `select_stage` reaches the ranker, and `mesh_actions` because
# it IS the arm: a screen that could load it could rank a query set by how much ADR-0022 helps.
FORBIDDEN = (
    "biolit.query.ranking",
    "biolit.query.concepts",
    "biolit.canon.mesh_actions",
    "biolit.canon.mesh_tree",
    "biolit.pipeline.stages",
)


def test_the_screen_cannot_reach_the_ranker_even_transitively():
    """⭐ BLINDNESS IS ENFORCED BY THE IMPORT GRAPH, NOT BY INTENT.

    The screen picks which queries the ADR-0022 pass will run on. If that choice could see
    arm disagreement, the pass would be scored on a population selected for the effect it is
    meant to measure -- ADR-0015's failure exactly, arriving one script earlier than anyone
    would look for it. Selecting on yield is legitimate (a query with 3 clusters tests
    nothing); selecting on effect direction is not, and the two are about ten lines apart.

    A comment saying "do not import the ranker" is not a mechanism. This is: a fresh
    interpreter imports the screen and reports its ENTIRE loaded module set, so a forbidden
    import fails this test whether it is written directly in the screen or acquired three
    modules deep by something the screen legitimately needs.

    It also enforces the right METRIC for free. `select_stage`'s kept-count differs BETWEEN
    the arms -- 70 against 75 on the old corpus -- so screening on "clusters kept" would
    already be screening on a quantity the arm changes. A screen that cannot import the filter
    cannot compute that number, so the only cluster count available to it is the raw
    pre-filter one, which is arm-independent by construction.
    """
    probe = (
        "import sys, json; import biolit_evals.relevance_screen; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('biolit'))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    loaded = json.loads(result.stdout)

    assert loaded, "the probe must actually have imported the screen"
    assert [m for m in FORBIDDEN if m in loaded] == []


def _row(slug: str, stratum: str, n_clusters: int) -> ScreenRow:
    return ScreenRow(slug, f"q {slug}", stratum, 40, 25, n_clusters)


def test_the_quota_keeps_the_separation_stratum_a_pure_top_k_would_delete():
    """⭐ THE ONE PLACE YIELD IS DELIBERATELY NOT THE CRITERION, stated here because it is the
    exception the pre-registration has to name.

    Stratum III is the structural drug classes -- fluoroquinolones, aminoglycosides -- where
    the MeSH tree connects a member to its class and the pharmacological-action relation does
    not. It is the ONLY population where ADR-0022 is inert while ADR-0020's tree term still
    fires, which makes it the clean separation case for isolating one from the other. It is
    also systematically less productive than the action-class queries, so top-k on yield
    deletes precisely the stratum that isolates the thing being measured.

    The fixture makes that concrete: every stratum-III candidate yields fewer clusters than
    every stratum-I one, so an unquota'd selection returns no III at all.
    """
    rows = [
        _row("i_a", "I", 20),
        _row("i_b", "I", 18),
        _row("i_c", "I", 16),
        _row("i_d", "I", 14),
        _row("iii_a", "III", 12),
        _row("iii_b", "III", 10),
        _row("thin", "III", 7),
    ]

    unquota = [r.slug for r in select(rows, k=4, quota={})]
    quota = [r.slug for r in select(rows, k=4, quota={"III": 2})]

    assert unquota == ["i_a", "i_b", "i_c", "i_d"], "pure yield deletes the separation case"
    assert set(quota) == {"i_a", "i_b", "iii_a", "iii_b"}
    assert "thin" not in quota, "MIN_CLUSTERS still applies inside the quota; it is not a bypass"


def test_screen_hash_pins_the_yield_facts_a_selection_cannot_claim_an_unrun_screen():
    """The freeze. The pre-registration commits this hash BEFORE any arm is scored, and the
    scorer refuses a selection whose screen does not reproduce it.

    ⚠️ It covers the CANDIDATE population, not just the chosen queries. A screen quietly re-run
    with three duds removed would select the same eight and look identical from the selection
    alone, while having changed what "the most productive eight of twenty-five" means. Only
    hashing the whole input catches that.

    Order-independent, because the candidate list is a set and a reordered rerun is the same
    screen. Sensitive to any yield number, because a re-run that changes a cluster count HAS
    moved the population underneath the freeze.
    """
    # Seven reverse-inserted elements, per this repo's determinism-fixture convention.
    rows = [_row(f"q{i}", "I" if i % 2 else "III", 20 - i) for i in range(7)]
    moved = [*rows[:-1], _row("q6", "I" if 6 % 2 else "III", 99)]

    assert screen_hash(rows) == screen_hash(list(reversed(rows)))
    assert screen_hash(rows) != screen_hash(rows[:-1]), "a dropped candidate is a different screen"
    assert screen_hash(rows) != screen_hash(moved), "a changed yield is a different screen"
