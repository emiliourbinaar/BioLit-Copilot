import pytest

from biolit.canon.mesh_tree import MeshTree, build_tree_table

_XML = """<?xml version="1.0"?>
<DescriptorRecordSet>
 <DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D003866</DescriptorUI>
  <DescriptorName><String>Depressive Disorder</String></DescriptorName>
  <SeeRelatedList>
   <SeeRelatedDescriptor><DescriptorReferredTo>
    <DescriptorUI>D000275</DescriptorUI>
   </DescriptorReferredTo></SeeRelatedDescriptor>
  </SeeRelatedList>
  <TreeNumberList><TreeNumber>F03.600.300</TreeNumber></TreeNumberList>
 </DescriptorRecord>
 <DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D063806</DescriptorUI>
  <DescriptorName><String>Myalgia</String></DescriptorName>
  <TreeNumberList>
   <TreeNumber>C05.651.542</TreeNumber>
   <TreeNumber>C23.888.592.612</TreeNumber>
  </TreeNumberList>
 </DescriptorRecord>
 <DescriptorRecord DescriptorClass="1">
  <DescriptorUI>D000068</DescriptorUI>
  <DescriptorName><String>No Tree Concept</String></DescriptorName>
 </DescriptorRecord>
</DescriptorRecordSet>
"""


def test_build_reads_the_records_own_ui_not_a_nested_cross_reference():
    """`<DescriptorUI>` also appears inside `SeeRelatedList/DescriptorReferredTo`. Reading the
    record's DIRECT child is unambiguous; a scan for the first match in the record happens to
    give the same answer only because the own-UI is emitted first, which is luck rather than a
    guarantee -- and D000275 must never acquire D003866's tree numbers."""
    table = build_tree_table(_XML)

    assert table["MESH:D003866"] == ["F03.600.300"]
    assert "MESH:D000275" not in table


def test_build_keeps_every_tree_number_for_a_concept_in_several_places():
    """Myalgia is filed under both muscular disease and the C23 signs-and-symptoms tree. Both
    matter: `distance` minimises over placements, so dropping one silently lengthens a
    distance and can flip a ranking."""
    assert build_tree_table(_XML)["MESH:D063806"] == ["C05.651.542", "C23.888.592.612"]


def test_build_omits_a_descriptor_that_carries_no_tree_number():
    """Not every descriptor is in the hierarchy. Storing an empty list would make
    `distance` unable to distinguish "not in the tree" from "not in the artifact"."""
    assert "MESH:D000068" not in build_tree_table(_XML)


def _tree() -> MeshTree:
    return MeshTree(
        {
            "MESH:PARENT": ["C10.228"],
            "MESH:CHILD": ["C10.228.140"],
            "MESH:GRANDCHILD": ["C10.228.140.163"],
            "MESH:SIBLING": ["C10.228.662"],
            "MESH:ELSEWHERE": ["F03.600"],
            "MESH:MULTI": ["F03.900", "C10.228.140.900"],
        }
    )


def test_distance_counts_edges_through_the_nearest_common_ancestor():
    tree = _tree()
    assert tree.distance("MESH:PARENT", "MESH:CHILD") == 1
    assert tree.distance("MESH:PARENT", "MESH:GRANDCHILD") == 2
    assert tree.distance("MESH:CHILD", "MESH:SIBLING") == 2
    assert tree.distance("MESH:CHILD", "MESH:CHILD") == 0


def test_distance_is_symmetric():
    tree = _tree()
    assert tree.distance("MESH:PARENT", "MESH:GRANDCHILD") == tree.distance(
        "MESH:GRANDCHILD", "MESH:PARENT"
    )


def test_distance_is_none_when_the_concepts_share_no_tree():
    """ADR-0020 relies on this being the common case for genuinely unrelated concepts: it is
    what puts `Acne Vulgaris` below the psychiatric clusters for a depression query. It must
    be None rather than a large number, so the caller sorts it last explicitly instead of
    competing on a magnitude that means something different."""
    assert _tree().distance("MESH:PARENT", "MESH:ELSEWHERE") is None


def test_distance_minimises_over_every_pair_of_placements():
    """MULTI sits in two trees, F03.900 and C10.228.140.900. Only the second is related to
    CHILD (C10.228.140), and it is a direct child of it, so the answer is 1. Taking MULTI's
    FIRST placement instead of the best would report None -- no relationship at all -- which
    is the failure this pins: a multiply-filed concept must not be judged by whichever tree
    the source file happened to list first."""
    assert _tree().distance("MESH:CHILD", "MESH:MULTI") == 1
    assert _tree().distance("MESH:CHILD", "MESH:ELSEWHERE") is None


def test_distance_is_none_for_a_concept_not_in_the_artifact():
    """An OMIM id, or a descriptor with no tree numbers. Ranking must degrade to "unknown",
    never raise: this runs on every cluster of every query."""
    tree = _tree()
    assert tree.distance("MESH:PARENT", "OMIM:125853") is None
    assert tree.distance("MESH:MISSING", "MESH:ALSO_MISSING") is None


def test_a_top_level_letter_alone_is_not_a_shared_ancestor():
    """ "C10" and "C23" are different top-level trees; a character-prefix comparison would
    call them related. Comparison is per DOT-SEPARATED NODE, and the first nodes differ."""
    tree = MeshTree({"MESH:A": ["C10.228"], "MESH:B": ["C23.888"]})
    assert tree.distance("MESH:A", "MESH:B") is None


def test_round_trips_through_the_artifact(tmp_path):
    table = build_tree_table(_XML)
    path = tmp_path / "tree.json.gz"
    MeshTree(table).save_artifact(str(path))

    assert MeshTree.from_artifact(str(path)).distance("MESH:D063806", "MESH:D063806") == 0


def test_build_refuses_a_document_with_no_descriptor_records():
    """Fail loud on a truncated or wrong-format download rather than silently producing an
    empty tree, which would make every distance None and every ranking degrade to key order
    with nothing saying so."""
    with pytest.raises(ValueError, match="no descriptor"):
        build_tree_table("<?xml version='1.0'?><DescriptorRecordSet></DescriptorRecordSet>")
