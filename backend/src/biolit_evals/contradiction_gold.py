from dataclasses import dataclass

from biolit.domain.records import ContradictionLabel
from biolit_evals.ctd_directions import Direction


@dataclass(frozen=True)
class GoldPair:
    """One manifest row. `chemical_id`/`disease_id` are the SHARED endpoints.

    For contradiction and agreement both are set (the pair shares a curated key). For
    insufficient_overlap exactly one is set -- the endpoint the two papers have in
    common -- because there IS no curated key joining them; that is what the class means.
    """

    paper_id_a: str
    paper_id_b: str
    chemical_id: str | None
    disease_id: str | None
    label: ContradictionLabel
    direction_a: str | None
    direction_b: str | None


def label_for_directions(
    a: frozenset[Direction], b: frozenset[Direction]
) -> ContradictionLabel | None:
    """Label a co-keyed pair, or None when it is not usable as gold.

    None is returned for a paper carrying BOTH directions on one key. CTD contains zero such
    papers (measured over all 109,591 direct-evidence rows), so this is unreachable on the
    real artifact -- but it is reachable BY TYPE, and returning None makes an unusable pair
    drop out of sampling instead of being silently labelled by whichever branch it fell into.
    """
    if len(a) != 1 or len(b) != 1:
        return None
    return ContradictionLabel.agreement if a == b else ContradictionLabel.contradiction
