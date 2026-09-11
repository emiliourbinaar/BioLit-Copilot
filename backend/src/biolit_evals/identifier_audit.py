"""DEF-0008 audit: which papers in the frozen corpora were identified or licensed wrongly.

Every frozen `--json-out` state records, per paper, the PMC id its licence was ACTUALLY looked up
under (`raw.pmc_id`). DEF-0008 made that a cited reference's id whenever the paper had none of
its own. This module compares each stored paper against its OWN identifiers, fetched fresh from
PubMed and read with the fixed parser, and traces every wrongly-allowed paper into what was built
on it -- records, clusters, quoted answers.

⚠️ READ-ONLY WITH RESPECT TO EVERY PUBLISHED NUMBER. It reports; it corrects nothing. Whether a
published figure changes is a decision taken on its report, not by running it.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from biolit.clients.pmc import normalize_pmcid

_CITED_PMID = re.compile(r"PMID (\d+)")


@dataclass(frozen=True)
class PaperVerdict:
    pmid: str
    stored_id: str
    doi_is_own: bool
    #: True when the licence lookup used this paper's own PMC id -- or, with no PMC id of its
    #: own, no lookup was made at all.
    licence_is_own: bool
    wrongly_allowed: bool


def _pmc(value: str | None) -> str | None:
    return normalize_pmcid(value) if value else None


def paper_verdict(
    stored: dict[str, Any], *, own_doi: str | None, own_pmc: str | None
) -> PaperVerdict:
    looked_up = _pmc((stored.get("raw") or {}).get("pmc_id"))
    licence_is_own = looked_up == _pmc(own_pmc)
    return PaperVerdict(
        pmid=str(stored.get("pmid") or ""),
        stored_id=str(stored["id"]),
        doi_is_own=stored.get("doi") == own_doi,
        licence_is_own=licence_is_own,
        wrongly_allowed=bool(stored.get("extraction_allowed")) and not licence_is_own,
    )


@dataclass(frozen=True)
class StateAudit:
    query: str
    n_candidates: int
    n_allowed_stored: int
    wrongly_allowed_pmids: list[str]
    n_doi_not_own: int
    #: Groups of PMIDs that shared one stored `Paper.id` -- each group is a DEF-0007 collapse.
    stored_id_collisions: list[list[str]]
    records_on_wrong: list[str]
    #: (cluster key, papers in it, of which wrongly allowed)
    clusters_on_wrong: list[tuple[str, int, int]]
    answer_cites_wrong: list[str]
    #: False where the state stores no answer text: `answer_cites_wrong` is then "nothing to
    #: look at", not a clean result. The frozen-8 states are like this; Gate A rendered later.
    answer_present: bool
    #: PMIDs PubMed no longer returns. Not judged, and so not counted as clean either.
    unverifiable_pmids: list[str]


def audit_state(
    state: Mapping[str, Any], own: Mapping[str, tuple[str | None, str | None]]
) -> StateAudit:
    """`own` maps PMID -> (own DOI, own PMC id), as the fixed parser reads them from PubMed."""
    papers = state["candidate_papers"]
    unverifiable = sorted(str(p["pmid"]) for p in papers if str(p["pmid"]) not in own)
    verdicts = [
        paper_verdict(p, own_doi=own[str(p["pmid"])][0], own_pmc=own[str(p["pmid"])][1])
        for p in papers
        if str(p["pmid"]) in own
    ]
    wrong = [v for v in verdicts if v.wrongly_allowed]
    wrong_ids = {v.stored_id for v in wrong}

    by_id: dict[str, list[str]] = {}
    for v in verdicts:
        by_id.setdefault(v.stored_id, []).append(v.pmid)

    records = state.get("extracted_records") or {}
    record_ids = set(records) if isinstance(records, Mapping) else {r["paper_id"] for r in records}
    clusters = [
        (c["key"], len(c["paper_ids"]), sum(pid in wrong_ids for pid in c["paper_ids"]))
        for c in state.get("clusters") or []
    ]
    cited = set(_CITED_PMID.findall(state.get("answer") or ""))

    return StateAudit(
        query=str(state.get("question", "")),
        n_candidates=len(papers),
        n_allowed_stored=sum(bool(p.get("extraction_allowed")) for p in papers),
        wrongly_allowed_pmids=sorted(v.pmid for v in wrong),
        n_doi_not_own=sum(not v.doi_is_own for v in verdicts),
        stored_id_collisions=sorted(sorted(g) for g in by_id.values() if len(g) > 1),
        records_on_wrong=sorted(wrong_ids & record_ids),
        clusters_on_wrong=[c for c in clusters if c[2] > 0],
        answer_cites_wrong=sorted(v.pmid for v in wrong if v.pmid in cited),
        answer_present=bool((state.get("answer") or "").strip()),
        unverifiable_pmids=unverifiable,
    )


#: Every corpus built through `PubMedClient.efetch` -- the path DEF-0008 was in. Corpora built
#: with `efetch_abstracts` (contradiction, Alamri) never read a DOI or PMC id and are out of scope.
CORPORA: dict[str, str] = {
    "frozen-8": "data/synth/states",
    "screen-25": "data/relevance2/states",
    "scale-3": "data/relevance2/scale",
}
RUN_LOG = "evals/identifier_audit_runs.jsonl"


def main() -> None:
    """Fetch each stored paper's own identifiers and audit every state. Run from `backend/`.

    No direct unit test, per project convention; `paper_verdict` and `audit_state` carry the
    logic and are tested in their own right.
    """
    import asyncio
    import json
    import xml.etree.ElementTree as ET
    from dataclasses import asdict
    from datetime import UTC, datetime
    from pathlib import Path

    import httpx

    from biolit.clients.pubmed import PubMedClient, _own_id, _pmc_id_of
    from biolit.config import get_settings

    async def own_identifiers(pmids: list[str]) -> dict[str, tuple[str | None, str | None]]:
        out: dict[str, tuple[str | None, str | None]] = {}
        async with httpx.AsyncClient(timeout=60) as http:
            client = PubMedClient(http, get_settings())
            for start in range(0, len(pmids), 200):
                chunk = pmids[start : start + 200]
                resp = await http.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params=client._params(db="pubmed", id=",".join(chunk), retmode="xml"),
                )
                resp.raise_for_status()
                for article in ET.fromstring(resp.text).findall("PubmedArticle"):
                    pmid = article.findtext("MedlineCitation/PMID") or ""
                    out[pmid] = (_own_id(article, "doi"), _pmc_id_of(article))
                await asyncio.sleep(0.4)
        return out

    ran_at = datetime.now(UTC).isoformat()
    log = Path(RUN_LOG)
    with log.open("a", encoding="utf-8") as handle:
        for corpus, directory in CORPORA.items():
            states = {
                path.name: json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(Path(directory).glob("*.json"))
            }
            pmids = sorted({str(p["pmid"]) for s in states.values() for p in s["candidate_papers"]})
            own = asyncio.run(own_identifiers(pmids))
            # A PMID PubMed no longer returns (withdrawn, merged, or a book record) cannot be
            # judged. `audit_state` lists those as unverifiable rather than clean; stopping the
            # whole audit for one would report nothing about the other 35 states.
            missing = [pmid for pmid in pmids if pmid not in own]
            if missing:
                print(f"{corpus}: {len(missing)} PMIDs have no PubMed record: {missing[:5]}")
            for name, state in states.items():
                audit = audit_state(state, own)
                handle.write(
                    json.dumps({"ran_at": ran_at, "corpus": corpus, "state": name, **asdict(audit)})
                    + "\n"
                )
                counts = (
                    audit.n_allowed_stored,
                    len(audit.wrongly_allowed_pmids),
                    audit.n_doi_not_own,
                    len(audit.stored_id_collisions),
                    len(audit.clusters_on_wrong),
                    len(audit.answer_cites_wrong),
                )
                print(f"{corpus:9} {name:40} " + " ".join(f"{n:4}" for n in counts))


if __name__ == "__main__":
    main()
