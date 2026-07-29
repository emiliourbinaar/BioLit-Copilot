from collections.abc import Sequence
from typing import Protocol

from biolit.domain.enums import EntityLabel
from biolit.domain.records import Entity


class PairingStrategy(Protocol):
    """Turns one paper's entities into (chemical_id, disease_id) pairs.

    Injected keyword-only into `cluster_papers`, mirroring the `Linker` seam: a real CID
    relation extractor later becomes a constructor argument, not a rewrite.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]: ...


def _linked_ids(entities: Sequence[Entity], label: EntityLabel) -> set[str]:
    """Canonical ids of one label. NIL entities are excluded: a `nil:<surface>` endpoint
    is unscoreable against gold CID, so admitting one would inject an unmeasurable
    population into a measurement whose whole purpose is precision. The cost of that
    exclusion is reported by `pairing_diagnostics`, not assumed away."""
    return {e.canonical_id for e in entities if e.label is label and e.canonical_id is not None}


class CrossProductPairing:
    """Every linked chemical paired with every linked disease in the same paper.

    The naive baseline. On gold entities it cannot miss a gold pair -- both endpoints are
    annotated, so the cross-product necessarily contains every gold pair, which is why
    key recall is 1.0 by construction (anchor 1). Its entire error is precision.
    """

    def pairs(self, entities: Sequence[Entity], text: str) -> set[tuple[str, str]]:
        chemicals = _linked_ids(entities, EntityLabel.CHEMICAL)
        diseases = _linked_ids(entities, EntityLabel.DISEASE)
        return {(c, d) for c in chemicals for d in diseases}
