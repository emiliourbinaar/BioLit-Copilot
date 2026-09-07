from biolit.canon.mesh import LinkResult, MeshConcept
from biolit.canon.mesh_actions import PharmacologicalActions
from biolit.domain.records import Cluster
from biolit.query.concepts import cluster_matches, resolve_query_concepts

_KNOWN = {
    "depressive disorder": MeshConcept(id="MESH:D003866", name="Depressive Disorder"),
    "isotretinoin": MeshConcept(id="MESH:D015474", name="Isotretinoin"),
    "metformin": MeshConcept(id="MESH:D008687", name="Metformin"),
    "atorvastatin": MeshConcept(id="MESH:D000069059", name="Atorvastatin"),
    "hydroxymethylglutaryl-coa reductase inhibitors": MeshConcept(
        id="MESH:D019161", name="Hydroxymethylglutaryl-CoA Reductase Inhibitors"
    ),
}

#: Atorvastatin's real declared action. The class carries no entry of its own, which is what
#: makes the reverse direction impossible rather than merely disallowed.
_ACTIONS = PharmacologicalActions({"MESH:D000069059": ["MESH:D019161"]})
_NO_ACTIONS = PharmacologicalActions({})


def _lookup(surface: str) -> LinkResult:
    return LinkResult(_KNOWN.get(surface), False)


def test_resolve_keeps_the_concept_the_local_dictionary_could_not_reach():
    """The point of routing query linking through NCBI. `depression` NILs against the local
    alias table in all 26 isotretinoin papers that contain it, so a query-side resolver built
    on the dictionary alone returns nothing for the disease half of the question."""
    concepts = resolve_query_concepts(("depression", "depressive disorder"), lookup=_lookup)

    assert "MESH:D003866" in concepts.ids
    assert concepts.evidence["MESH:D003866"] == "depressive disorder"


def test_resolve_records_the_terms_it_could_not_link_rather_than_discarding_them():
    """`unresolved` is diagnostic, not decorative: a query whose disease half resolves to
    nothing behaves identically to one with no disease half at all, and only this field
    distinguishes them in the ledger."""
    concepts = resolve_query_concepts(("metformin", "wibble"), lookup=_lookup)

    assert concepts.ids == frozenset({"MESH:D008687"})
    assert concepts.unresolved == ("wibble",)


def test_resolve_reports_an_empty_concept_set_when_nothing_links():
    """The fail-open trigger. It must be an ordinary value, not an exception -- an
    unlinkable query is a normal event and must not take the pipeline down."""
    concepts = resolve_query_concepts(("wibble", "wobble"), lookup=_lookup)

    assert concepts.ids == frozenset()
    assert concepts.unresolved == ("wibble", "wobble")


def test_a_cluster_matches_on_either_side_of_its_key():
    """Deliberately an OR, not an AND. Requiring both sides would drop `Aspirin | Hemorrhage`
    from an NSAIDs/bleeding query -- a clinically adjacent comparator the reader wants -- and
    on the eight frozen queries an AND keeps almost nothing."""
    concepts = resolve_query_concepts(("metformin",), lookup=_lookup)

    def match(key: str) -> bool:
        return cluster_matches(Cluster(key=key, paper_ids=["a"]), concepts, actions=_NO_ACTIONS)

    assert match("MESH:D008687|MESH:D000140")
    assert match("MESH:D000140|MESH:D008687")
    assert not match("MESH:D000140|MESH:D006470")


def test_a_cluster_never_matches_an_empty_concept_set():
    """`cluster_matches` answers "does this cluster overlap the query", and the honest answer
    for a query that resolved to nothing is no. Fail-open belongs in the STAGE, which can say
    so in the ledger; hiding it here would make an unfiltered run indistinguishable from a
    filtered one that happened to keep everything."""
    concepts = resolve_query_concepts(("wibble",), lookup=_lookup)

    assert not cluster_matches(
        Cluster(key="MESH:D008687|MESH:D000140", paper_ids=["a"]), concepts, actions=_NO_ACTIONS
    )


def test_a_class_query_matches_a_member_but_a_member_query_never_matches_the_class():
    """ADR-0022, and the direction is the decision -- so it is pinned, not commented.

    Forward: "statins and rhabdomyolysis" resolves to the CLASS D019161, and the three
    clusters DEF-0002 named carry the MEMBER Atorvastatin. Nothing in the MeSH tree connects
    them -- the member sits in D03/D10 (chemical structure), the class in D27 (actions and
    uses) -- so before this the three were dropped as off-query.

    ⚠️ Reverse: a query resolving to `Atorvastatin` must NOT pull in every other statin by way
    of the shared class. Only member->class has a demonstrated consumer (ADR-0013), and the
    asymmetry is structural rather than guarded: the lookup is keyed on the CLUSTER side, so
    the class descriptor -- which declares no action of its own -- has nothing to match with.
    """
    statins = resolve_query_concepts(
        ("hydroxymethylglutaryl-coa reductase inhibitors",), lookup=_lookup
    )
    atorvastatin = resolve_query_concepts(("atorvastatin",), lookup=_lookup)

    member_cluster = Cluster(key="MESH:D000069059|MESH:D009135", paper_ids=["a"])
    class_cluster = Cluster(key="MESH:D019161|MESH:D009135", paper_ids=["a"])

    assert cluster_matches(member_cluster, statins, actions=_ACTIONS)
    assert not cluster_matches(class_cluster, atorvastatin, actions=_ACTIONS)
