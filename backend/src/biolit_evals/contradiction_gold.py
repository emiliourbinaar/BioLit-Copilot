import itertools
from collections import defaultdict
from collections.abc import Iterator, Mapping
from collections.abc import Set as AbstractSet
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


def build_candidates(
    directions: Mapping[str, Mapping[tuple[str, str], frozenset[Direction]]],
    *,
    excluded: AbstractSet[str],
    max_per_key: int = 4,
) -> list[GoldPair]:
    """Invert pmid-major CTD directions into key-major candidate pairs.

    `excluded` pmids (BC5CDR training pmids) are dropped before any pairing happens, so they
    can never appear in a candidate. `max_per_key` bounds how many pairs of EACH label are
    emitted per key -- real CTD has ~2.78M agreeing co-keyed pairs, and `sample_pairs` only
    ever takes one pair per key, so materialising more than a handful per key is pure waste.
    Iteration is sorted throughout, so output order is deterministic before any shuffle.
    """
    by_key: dict[tuple[str, str], dict[str, frozenset[Direction]]] = defaultdict(dict)
    for pmid, keys in directions.items():
        if pmid in excluded:
            continue
        for key, dirs in keys.items():
            by_key[key][pmid] = dirs

    out: list[GoldPair] = []
    for key in sorted(by_key):
        pmid_dirs = by_key[key]
        pmids = sorted(pmid_dirs)
        chemical_id, disease_id = key
        emitted_by_label: dict[ContradictionLabel, int] = defaultdict(int)
        full_labels: set[ContradictionLabel] = set()
        for a, b in itertools.combinations(pmids, 2):
            if len(full_labels) >= 2:
                break
            label = label_for_directions(pmid_dirs[a], pmid_dirs[b])
            if label is None or label in full_labels:
                continue
            dir_a = next(iter(pmid_dirs[a])).value
            dir_b = next(iter(pmid_dirs[b])).value
            out.append(GoldPair(a, b, chemical_id, disease_id, label, dir_a, dir_b))
            emitted_by_label[label] += 1
            if emitted_by_label[label] >= max_per_key:
                full_labels.add(label)

    by_chemical: dict[str, set[str]] = defaultdict(set)
    by_disease: dict[str, set[str]] = defaultdict(set)
    pmid_keys: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key, pmid_dirs in by_key.items():
        chem, dis = key
        for pmid in pmid_dirs:
            by_chemical[chem].add(pmid)
            by_disease[dis].add(pmid)
            pmid_keys[pmid].add(key)

    def hard_negatives(
        endpoint_map: Mapping[str, set[str]], *, is_chemical: bool
    ) -> Iterator[GoldPair]:
        for endpoint in sorted(endpoint_map):
            pmids = sorted(endpoint_map[endpoint])
            emitted = 0
            for a, b in itertools.combinations(pmids, 2):
                if emitted >= max_per_key:
                    break
                if pmid_keys[a] & pmid_keys[b]:
                    continue
                yield GoldPair(
                    a,
                    b,
                    endpoint if is_chemical else None,
                    None if is_chemical else endpoint,
                    ContradictionLabel.insufficient_overlap,
                    None,
                    None,
                )
                emitted += 1

    out.extend(hard_negatives(by_chemical, is_chemical=True))
    out.extend(hard_negatives(by_disease, is_chemical=False))
    return out
