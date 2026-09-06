"""MeSH descriptor tree numbers, and the distance between two concepts in the hierarchy.

Built for ADR-0020's ranking score and justified by that consumer alone (ADR-0013). Source is
`data/mesh/desc2026.gz`, already on disk from the ADR-0011 work -- 31,108 descriptors carry
tree numbers -- so this adds a build step, not a download.

⚠️ THIS IS TREE DISTANCE, NOT CLINICAL RELATEDNESS, and ADR-0020 records it as unfit to
filter on for that reason. It errs in both directions: `C23` ("Pathological Conditions, Signs
and Symptoms") is a grab-bag that manufactures false proximity -- `Hemorrhage` to `Atrial
Fibrillation` measures 3 -- while clinically related concepts filed under different trees
measure None, as `Stroke` does from `Hemorrhage`. A ranking error demotes a cluster; a filter
error deletes it, which is why this signal is confined to ordering.
"""

import gzip
import json
from collections.abc import Mapping
from xml.etree import ElementTree as ET


def build_tree_table(xml_text: str) -> dict[str, list[str]]:
    """Parse MeSH descriptor XML into `{"MESH:D003866": ["F03.600.300"], ...}`.

    Reads each record's DIRECT `DescriptorUI` child. `<DescriptorUI>` also occurs nested in
    `SeeRelatedList/SeeRelatedDescriptor/DescriptorReferredTo`, so a scan for the first match
    inside a record gives the right answer only because the own-UI happens to be emitted
    first -- luck, not a guarantee, and getting it wrong would attribute one concept's tree
    numbers to an unrelated cross-referenced one.

    A descriptor with no tree numbers is OMITTED rather than stored empty, so `distance` can
    treat "absent" as one thing: no usable placement.
    """
    root = ET.fromstring(xml_text)
    records = root.findall("DescriptorRecord")
    if not records:
        raise ValueError(
            "build_tree_table: no descriptor records found. Refusing to build an empty tree "
            "-- every distance would be None and every ranking would silently degrade to key "
            "order with nothing in the output saying so. Check the download."
        )
    table: dict[str, list[str]] = {}
    for record in records:
        ui = record.findtext("DescriptorUI")
        if not ui:
            continue
        numbers = [
            number.text for number in record.findall("TreeNumberList/TreeNumber") if number.text
        ]
        if numbers:
            table[f"MESH:{ui}"] = numbers
    return table


class MeshTree:
    """Concept id -> its MeSH tree numbers, with a distance over the hierarchy."""

    def __init__(self, numbers: Mapping[str, list[str]]) -> None:
        self._numbers = dict(numbers)

    def __len__(self) -> int:
        return len(self._numbers)

    def distance(self, a: str, b: str) -> int | None:
        """Edges between `a` and `b` via their nearest common ancestor, or None.

        None means "no shared placement anywhere in the hierarchy", and covers three cases
        that are the same for a caller: different top-level trees, a concept with no tree
        numbers, and an id that is not a MeSH descriptor at all (an `OMIM:` concept). It is
        deliberately NOT a large integer -- a caller must sort it last as a distinct
        category rather than let it compete on a magnitude that means something else.

        Minimised over every pair of placements, because a concept may sit in several trees:
        `Myalgia` is filed under both muscular disease and C23, and taking only the first
        would report no relationship where a close one exists.
        """
        left, right = self._numbers.get(a), self._numbers.get(b)
        if not left or not right:
            return None
        best: int | None = None
        for one in left:
            for other in right:
                shared = _shared_nodes(one, other)
                if shared == 0:
                    continue
                edges = (one.count(".") + 1 - shared) + (other.count(".") + 1 - shared)
                best = edges if best is None else min(best, edges)
        return best

    def save_artifact(self, path: str) -> None:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(self._numbers, fh)

    @classmethod
    def from_artifact(cls, path: str) -> "MeshTree":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls(json.load(fh))


def _shared_nodes(one: str, other: str) -> int:
    """Count of leading DOT-SEPARATED nodes the two tree numbers share.

    Per node, never per character: "C10.228" and "C23.888" share the letter C and nothing
    else, and a string-prefix comparison would call two different top-level trees related.
    """
    left, right = one.split("."), other.split(".")
    shared = 0
    for node_a, node_b in zip(left, right, strict=False):
        if node_a != node_b:
            break
        shared += 1
    return shared
