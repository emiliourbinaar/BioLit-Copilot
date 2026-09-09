"""Yield-only query screen for the ADR-0022 validation pass.

⚠️ THIS MODULE IS DELIBERATELY BLIND, AND THE BLINDNESS IS STRUCTURAL. It decides which
queries the pass runs on, so if it could see how much the arms disagree, the pass would be
scored on a population selected for the effect it is meant to measure. That is ADR-0015's
"an arm scored on a population it cannot lose on proves nothing", arriving one script earlier
than anyone would think to look.

⭐ THE MECHANISM IS THE IMPORT GRAPH, NOT A PROMISE. This module imports NOTHING from
`biolit` -- not the ranker, not the filter, not the MeSH artifacts, not even `Cluster`. It
reads the saved pipeline state as plain JSON and counts. `tests/evals/test_relevance_screen.py`
spawns a fresh interpreter, imports this module, and asserts the ranker is absent from the
resulting `sys.modules`, so a forbidden import fails CI whether written here directly or
acquired transitively.

⭐ AND IT PICKS THE METRIC FOR FREE. `select_stage`'s kept-count DIFFERS BETWEEN THE ARMS --
70 against 75 on the old corpus -- so screening on "clusters kept" would already be screening
on a quantity the arm changes. A module that cannot import the filter cannot compute that
number. The only cluster count reachable from here is the raw pre-filter one, which is
arm-independent by construction.

Selecting on YIELD is legitimate: a query producing three clusters tests nothing, whichever
arm is right. Selecting on DIRECTION is not. This module can only do the first.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: A query below this contributes almost no rankable pairs. Fixed here, before any candidate
#: has been run, so it cannot drift toward whatever the observed distribution happens to be.
MIN_CLUSTERS = 8


@dataclass(frozen=True)
class ScreenRow:
    """What is knowable about a candidate query without consulting either arm."""

    slug: str
    query: str
    stratum: str
    n_papers: int
    n_licensed: int
    n_clusters: int

    @property
    def n_pairs(self) -> int:
        """Rankable pairs, the quantity the pass actually spends. Quadratic in clusters, which
        is why a query yielding 3 and one yielding 11 are not 3.7x apart but 18x apart."""
        return self.n_clusters * (self.n_clusters - 1) // 2


def yield_row(slug: str, query: str, stratum: str, state: Mapping) -> ScreenRow:
    """Read one saved pipeline state into a screen row.

    Takes the state as an already-parsed mapping rather than a path so the caller owns all
    file access, and so this stays a pure function over data the test can construct.
    """
    stages = {stage["name"]: stage for stage in state["stages"]}
    return ScreenRow(
        slug=slug,
        query=query,
        stratum=stratum,
        n_papers=stages["retrieve"]["n_out"],
        n_licensed=stages["licence_gate"]["n_out"],
        # RAW clusters, pre-filter. See the module docstring: the filtered count is arm-dependent
        # and is not reachable from here.
        n_clusters=stages["cluster"]["n_out"],
    )


def select(rows: Sequence[ScreenRow], *, k: int, quota: Mapping[str, int]) -> list[ScreenRow]:
    """Pick the queries to run the pass on: the most productive, subject to a stratum quota.

    ⚠️ THE QUOTA IS WHY THIS IS NOT PURE YIELD-MAXIMISATION. Stratum III -- structural drug
    classes, where the MeSH tree connects a member to its class and the pharmacological-action
    relation does not -- is the only population where ADR-0022 is inert while ADR-0020's tree
    term still fires. That makes it the clean separation case for Design B, and it is also
    systematically less productive than the action-class queries, so a pure top-k would delete
    exactly the stratum that isolates the thing being measured.

    The quota is filled FIRST, from each stratum's own ranking, and the remaining slots go to
    the global ranking of whatever is left. Ties break on slug so a rerun selects identically.
    """
    eligible = sorted(
        (row for row in rows if row.n_clusters >= MIN_CLUSTERS),
        key=lambda row: (-row.n_clusters, row.slug),
    )
    chosen: list[ScreenRow] = []
    for stratum, want in sorted(quota.items()):
        chosen.extend([row for row in eligible if row.stratum == stratum][:want])
    for row in eligible:
        if len(chosen) >= k:
            break
        if row not in chosen:
            chosen.append(row)
    return sorted(chosen, key=lambda row: (-row.n_clusters, row.slug))[:k]


def screen_hash(rows: Sequence[ScreenRow]) -> str:
    """Pin the screen's INPUT so a later selection cannot claim a screen it did not run on.

    Order-independent and over the yield facts only, mirroring `rows_hash`/`labels_hash`. A
    candidate re-run that changes a cluster count changes this, which is the point: the frozen
    artifact must stop matching if the population underneath it moved.
    """
    payload = sorted(
        f"{row.slug}|{row.stratum}|{row.n_papers}|{row.n_licensed}|{row.n_clusters}" for row in rows
    )
    return hashlib.sha256("\n".join(payload).encode()).hexdigest()


def render_frozen(rows: Sequence[ScreenRow], chosen: Sequence[ScreenRow]) -> str:
    """The artifact committed BEFORE any arm is scored. Records what was screened, what was
    selected, and the hash the scorer must later match."""
    return json.dumps(
        {
            "min_clusters": MIN_CLUSTERS,
            "screen_hash": screen_hash(rows),
            "candidates": [
                {
                    "slug": r.slug,
                    "query": r.query,
                    "stratum": r.stratum,
                    "n_papers": r.n_papers,
                    "n_licensed": r.n_licensed,
                    "n_clusters": r.n_clusters,
                    "n_pairs": r.n_pairs,
                }
                for r in sorted(rows, key=lambda r: (-r.n_clusters, r.slug))
            ],
            "selected": [r.slug for r in chosen],
        },
        indent=2,
    )
