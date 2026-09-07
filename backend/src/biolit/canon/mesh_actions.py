"""MeSH PharmacologicalAction: the class a chemical belongs to, which is NOT a tree edge.

Built for ADR-0022's member-to-class match and justified by that consumer alone (ADR-0013).
Source is `data/mesh/desc2026.gz`, the same dump `build_mesh_tree` reads -- 2,838 of 31,108
descriptors declare an action -- so this adds a build step, not a download.

⚠️ WHY THIS EXISTS AT ALL, rather than reusing `MeshTree`. Measured: `Atorvastatin` is filed
under `D03.383.129.578.075` and `D10.251.450.200`, both chemical-structure trees, while
`Hydroxymethylglutaryl-CoA Reductase Inhibitors` is filed under `D27`, "Chemical Actions and
Uses". They share no node, so `MeshTree.distance` returns None between a statin and the statin
class. Walking the hierarchy cannot reach a drug class from its member because for drug classes
the relation is not in the hierarchy; it is this field.

⚠️ THE RELATION IS DIRECTED AND SO IS ITS USE. This maps member -> the classes it belongs to,
the direction ADR-0022 ships. The reverse index (class -> its members) is deliberately NOT
built: it has no demonstrated consumer, and materialising every member of a class is the
over-broad move ADR-0020 rejected for the filter.
"""

import gzip
import json
from collections.abc import Mapping
from xml.etree import ElementTree as ET

_ACTION_PATH = "PharmacologicalActionList/PharmacologicalAction/DescriptorReferredTo/DescriptorUI"


def build_action_table(xml_text: str) -> dict[str, list[str]]:
    """Parse MeSH descriptor XML into `{"MESH:D000069059": ["MESH:D019161"], ...}`.

    Reads each record's DIRECT `DescriptorUI` child for the key and the full path above for
    the values. Both matter, and the second is the subtler one: `DescriptorReferredTo/
    DescriptorUI` ALSO occurs under `SeeRelatedList`, so unlike `build_tree_table` -- where the
    payload element has a distinct tag -- the element to read and the element to ignore here
    are identical, separated only by their parent list.

    A descriptor declaring no action is OMITTED rather than stored empty, so `classes_of` can
    treat "absent" as one thing.
    """
    root = ET.fromstring(xml_text)
    records = root.findall("DescriptorRecord")
    if not records:
        raise ValueError(
            "build_action_table: no descriptor records found. Refusing to build an empty "
            "table -- every classes_of would return nothing, the match predicate would "
            "collapse silently back to exact-only, and the clusters ADR-0022 recovers would "
            "be dropped again with nothing in the output saying why. Check the download."
        )
    table: dict[str, list[str]] = {}
    for record in records:
        ui = record.findtext("DescriptorUI")
        if not ui:
            continue
        actions = [
            f"MESH:{element.text}" for element in record.findall(_ACTION_PATH) if element.text
        ]
        if actions:
            table[f"MESH:{ui}"] = actions
    return table


class PharmacologicalActions:
    """Concept id -> the MeSH classes it declares a pharmacological action for."""

    def __init__(self, actions: Mapping[str, list[str]]) -> None:
        self._actions = {key: frozenset(value) for key, value in actions.items()}

    def __len__(self) -> int:
        return len(self._actions)

    def classes_of(self, concept_id: str) -> frozenset[str]:
        """The classes `concept_id` belongs to, or an empty set.

        EMPTY RATHER THAN None for a miss, because the caller intersects this against the
        query concepts on every side of every cluster. Most of those sides are diseases, which
        declare no action, and a few are not MeSH descriptors at all -- an `OMIM:` id reaches
        this lookup like any other. None of those is an error; they are all "belongs to no
        class the query named".
        """
        return self._actions.get(concept_id, frozenset())

    def save_artifact(self, path: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump({key: sorted(value) for key, value in self._actions.items()}, fh)

    @classmethod
    def from_artifact(cls, path: str) -> "PharmacologicalActions":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls(json.load(fh))
