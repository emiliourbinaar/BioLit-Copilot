"""Resolve a user's question to MeSH concepts, and test a cluster against them.

WHY THE QUERY IS LINKED THROUGH NCBI RATHER THAN THE LOCAL DICTIONARY. Measured on
2026-09-05 against the eight frozen queries, the local alias table NILs on three of eight
disease-side terms -- `depression`, `gastrointestinal bleeding`, `thyroid dysfunction` --
and the failure is not confined to the query string: the bare word `depression` appears in
26 of the 60 isotretinoin papers and links to nothing in every one, which is *why* that run
produces no depression cluster at all. NCBI's own query translation resolves all three
("depressive disorder", "gastrointestinal hemorrhage", "thyroid gland"), rides free on the
search request the pipeline already makes, and needs no LLM.

⚠️ WHAT THIS DOES NOT FIX, measured rather than assumed. Overlap is an OR across both sides
of a cluster key, and the *chemical* side links locally for all eight queries, so NCBI's
extra reach changes no keep/drop decision on the frozen corpus: query-side linking through
NCBI and through the local dictionary both keep exactly 70 of 83 clusters, cluster for
cluster. The added reach pays off only for a consumer that needs the query's DISEASE concept
specifically -- ordering by topical distance, which is deferred -- not for this filter.

⚠️ AND WHAT THE FILTER ITSELF DOES NOT FIX. It keeps 70 of 83 clusters, so it prunes an
off-topic tail; it does not decide what an answer LEADS with. "isotretinoin and depression"
keeps 5 of 5 and still opens on `Isotretinoin | Acne Vulgaris` (18 papers). Leading is an
ordering property, `render_cluster` deliberately carries no ranking, and changing that
overturns a stated design decision -- so it is a separate call, not a patch to this module.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from biolit.canon.mesh import LinkResult
from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.domain.records import Cluster


@dataclass(frozen=True)
class QueryConcepts:
    """What the question resolved to, and what it did not.

    `unresolved` is load-bearing rather than diagnostic decoration: a query whose disease
    half links to nothing produces the same `ids` as a query that had no disease half, and
    without this field the stage ledger cannot tell a reader which of the two happened.
    """

    ids: frozenset[str]
    evidence: Mapping[str, str]
    unresolved: tuple[str, ...]


def resolve_query_concepts(
    concept_terms: Sequence[str], *, lookup: Callable[[str], LinkResult]
) -> QueryConcepts:
    """Link NCBI's translated concept terms against the MeSH dictionary.

    `lookup` is injected rather than taking a `MeshDictionary` so this stays free of the
    550k-alias artifact, matching how `entities_stage` takes `extract`.

    An empty result is an ordinary value, never an exception: an unlinkable query is a
    normal event, and the decision about what to do with it belongs to the stage, which can
    record the choice in the ledger.
    """
    ids: dict[str, str] = {}
    unresolved: list[str] = []
    for term in concept_terms:
        concept = lookup(term).concept
        if concept is None:
            unresolved.append(term)
        elif concept.id not in ids:
            ids[concept.id] = term
    return QueryConcepts(ids=frozenset(ids), evidence=dict(ids), unresolved=tuple(unresolved))


def side_matches(side: str, concepts: QueryConcepts, actions: PharmacologicalActions) -> bool:
    """True when this one side of a cluster key answers the query: it IS a query concept, or
    it belongs to a pharmacological class the query named (ADR-0022).

    ⚠️ MEMBER -> CLASS ONLY, and the asymmetry is structural rather than enforced. The lookup
    is keyed on the SIDE, so a cluster carrying `Atorvastatin` answers a question about
    HMG-CoA reductase inhibitors, while a question about Atorvastatin does not pull in the
    class -- the class descriptor declares no action of its own, so there is nothing to match
    on. The reverse direction has no demonstrated consumer (ADR-0013) and is not built.

    THE SAME PREDICATE ORDERS AS FILTERS. `_relevance_key` calls this too, and it must: a
    cluster kept only by the class relation but scored as an exact-match miss sorts BELOW
    every merely-background cluster. Measured on the frozen corpus, filtering on this while
    ranking on identity alone puts the three recovered statins clusters at positions 15-17 of
    17, under seven background ones.
    """
    return side in concepts.ids or bool(actions.classes_of(side) & concepts.ids)


def cluster_matches(
    cluster: Cluster, concepts: QueryConcepts, *, actions: PharmacologicalActions
) -> bool:
    """True when either side of the cluster key answers the query.

    DELIBERATELY AN OR. Requiring both sides drops `Aspirin | Hemorrhage` from an
    NSAIDs/bleeding question and `Lactic Acid | Acidosis` from a metformin/lactic-acidosis
    one -- the comparator and the mechanism, both of which a reader asking that question
    wants. On the frozen corpus an AND keeps almost nothing.

    Answers no for an empty concept set, which is the honest answer to "does this overlap
    the query" when the query resolved to nothing. Fail-open is the STAGE's decision, so
    that an unfiltered run stays distinguishable in the ledger from a filtered one that
    happened to keep everything.
    """
    return any(side_matches(side, concepts, actions) for side in cluster.key.split("|"))
