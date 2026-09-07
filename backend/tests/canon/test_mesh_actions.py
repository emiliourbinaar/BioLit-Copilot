import pytest

from biolit.canon.mesh_actions import PharmacologicalActions, build_action_table

# Real relations from desc2026, not invented: Atorvastatin declares both an
# Anticholesteremic Agents and an HMG-CoA Reductase Inhibitors action, and the class
# descriptor itself declares none while carrying a SeeRelatedList -- which is exactly the
# record shape that punishes a record-wide scan.
_XML = """<?xml version="1.0"?>
<DescriptorRecordSet>
 <DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000069059</DescriptorUI>
  <DescriptorName><String>Atorvastatin</String></DescriptorName>
  <PharmacologicalActionList>
   <PharmacologicalAction><DescriptorReferredTo>
    <DescriptorUI>D000924</DescriptorUI>
    <DescriptorName><String>Anticholesteremic Agents</String></DescriptorName>
   </DescriptorReferredTo></PharmacologicalAction>
   <PharmacologicalAction><DescriptorReferredTo>
    <DescriptorUI>D019161</DescriptorUI>
    <DescriptorName><String>Hydroxymethylglutaryl-CoA Reductase Inhibitors</String></DescriptorName>
   </DescriptorReferredTo></PharmacologicalAction>
  </PharmacologicalActionList>
 </DescriptorRecord>
 <DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D019161</DescriptorUI>
  <DescriptorName><String>Hydroxymethylglutaryl-CoA Reductase Inhibitors</String></DescriptorName>
  <SeeRelatedList>
   <SeeRelatedDescriptor><DescriptorReferredTo>
    <DescriptorUI>D008074</DescriptorUI>
   </DescriptorReferredTo></SeeRelatedDescriptor>
  </SeeRelatedList>
 </DescriptorRecord>
</DescriptorRecordSet>
"""


def test_build_reads_only_the_pharmacological_action_list_not_every_referred_descriptor():
    """⚠️ THE TRAP IS SHARPER HERE THAN IN `build_tree_table`. There, the element that must be
    ignored (`SeeRelatedDescriptor/.../DescriptorUI`) differs from the payload element
    (`TreeNumber`) by tag. Here the element to ignore and the element to read are BOTH
    `DescriptorReferredTo/DescriptorUI`, distinguished only by which list they hang under.

    D019161's see-related D008074 is `Mevalonic Acid` -- chemically adjacent, so a record-wide
    scan would produce a relation that looks plausible and is not asserted by MeSH at all.
    """
    table = build_action_table(_XML)

    assert table["MESH:D000069059"] == ["MESH:D000924", "MESH:D019161"]
    assert not any("MESH:D008074" in actions for actions in table.values())


def test_build_omits_a_descriptor_that_declares_no_action():
    """Most descriptors declare none -- 2,838 of 31,108 carry one. Storing an empty list would
    make `classes_of` unable to distinguish "declares no action" from "not in the artifact",
    and the whole predicate turns on that lookup being a single unambiguous thing.

    D019161 is the case that matters: it is the CLASS, and it is the concept the statins query
    itself resolves to. If a class could carry an entry, the member->class direction ADR-0022
    fixed would be one lookup away from silently becoming symmetric.
    """
    assert "MESH:D019161" not in build_action_table(_XML)


def test_build_refuses_a_dump_it_could_not_parse():
    """Mirrors `build_tree_table`. An empty table is not a degraded artifact, it is a silent
    reversal: every `classes_of` returns empty, the match predicate collapses back to
    exact-only, and the three clusters ADR-0022 recovers are dropped again -- with the run
    reporting a completed SELECT stage and a plausible cluster count either way.
    """
    with pytest.raises(ValueError, match="no descriptor records"):
        build_action_table('<?xml version="1.0"?><DescriptorRecordSet/>')


def test_classes_of_is_a_set_for_membership_and_empty_for_an_unknown_concept():
    """`side_matches` intersects this against the query concepts on EVERY side of EVERY
    cluster, so the miss case has to be an empty set rather than None. Most of those sides are
    diseases, which declare no action, and some are not MeSH descriptors at all -- an `OMIM:`
    id reaches this lookup like any other. None of those is an error condition.
    """
    actions = PharmacologicalActions(build_action_table(_XML))

    assert actions.classes_of("MESH:D000069059") == frozenset({"MESH:D000924", "MESH:D019161"})
    assert actions.classes_of("OMIM:143890") == frozenset()
    assert len(actions) == 1


def test_the_artifact_round_trips(tmp_path):
    """The runtime never parses the dump; it loads this. A membership set that survived
    `json.dump` as a list and came back as one would still answer `in` correctly and would
    quietly stop answering `&`, which is the operation the predicate actually performs.
    """
    path = str(tmp_path / "actions.json.gz")
    PharmacologicalActions(build_action_table(_XML)).save_artifact(path)
    reloaded = PharmacologicalActions.from_artifact(path)

    assert reloaded.classes_of("MESH:D000069059") & {"MESH:D019161"} == {"MESH:D019161"}
