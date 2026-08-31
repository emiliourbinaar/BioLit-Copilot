# End-to-End Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI that takes a query, runs it through the real Phase 1–5 components (PubMed retrieval → NER → canonicalization → deterministic extraction → clustering), and prints a per-stage report with an honest drop ledger, while marking the Critic and Synthesis stages as explicitly unimplemented.

**Architecture:** Fix the broken PMC licence lookup first (nothing downstream produces output until it works), then add one `stages` field to `PipelineState`, then a `biolit.pipeline` package of pure stage functions with a thin `__main__.py` CLI. Stage functions are pure and independently testable; only `__main__.py` touches the network or loads models.

**Tech Stack:** Python 3.12, uv, pydantic v2, httpx, respx (HTTP cassettes), pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-08-31-end-to-end-pipeline-design.md` — read it alongside this plan; every task argues from it.

## Global Constraints

- All commands run from `backend/` via `uv run`. Never from the repo root.
- Ruff ruleset `E,F,I,UP,B`; line length 100. Gate = `ruff check` + `ruff format --check` + `pyright` + `pytest`.
- String enums use `enum.StrEnum` (ADR-0005). Never `class X(str, Enum)`.
- `datetime.now(UTC)` — never `datetime.now(timezone.utc)`.
- Imports at the top of the file (E402). The **only** exception is deliberate function-local heavy imports inside `main()` (torch/transformers), matching `end_to_end.main`.
- **Unit tests must never download anything or touch the network.** Every HTTP interaction is a respx cassette under `backend/tests/cassettes/`.
- **tdd-guard is active.** Write the failing test first and run it. It blocks implementation without a failing test, and blocks over-implementation while a test still fails at import rather than at its assertion. If blocked, re-run tests to establish green, then proceed.
- **Never manufacture a fake RED** by deliberately writing wrong behaviour.
- `main()` gets no direct unit test. Every function it calls is tested in its own module.
- **Mutations are RUN, never reasoned about** (ADR-0016 rule 1). Verify restoration with a saved-copy `diff`, never `git diff` — `git diff` reports nothing for an untracked file.
- Commit messages: **no `Claude-Session:` trailer, no session URL, no agent attribution.** Public portfolio repo.
- Use `git commit -F -` with a bash heredoc. Never PowerShell here-strings.
- CPU-pin guard: `grep -ciE '^name = "(nvidia|triton)' uv.lock` must return 0.
- Do not commit `backend/data/`. Run logs are committed; **this sub-project adds no run log.**

## File Structure

| File | Responsibility |
|---|---|
| `src/biolit/domain/licensing.py` *(modify)* | Add `license_token_from_url` — URL → canonical token. Tier table untouched. |
| `src/biolit/clients/pmc.py` *(create)* | Pure parsing of PMC article XML: licence URL per PMCID. No HTTP. |
| `src/biolit/clients/pubmed.py` *(modify)* | Remove the dead OA lookup; batch permissions via `efetch db=pmc`; `_parse_article` becomes sync and pure. |
| `src/biolit/state/pipeline.py` *(modify)* | Add `StageStatus`, `StageReport`, and `PipelineState.stages`. |
| `src/biolit/pipeline/__init__.py` *(create)* | Empty package marker. |
| `src/biolit/pipeline/stages.py` *(create)* | Pure stage functions, each returning its result plus a `StageReport`. |
| `src/biolit/pipeline/report.py` *(create)* | Renders a `list[StageReport]` to the human-readable text block. |
| `src/biolit/pipeline/__main__.py` *(create)* | CLI. Heavy imports local to `main()`. No unit test. |

---

### Task 1: Licence URL → canonical token

**Files:**
- Modify: `backend/src/biolit/domain/licensing.py`
- Test: `backend/tests/domain/test_licensing.py`

**Interfaces:**
- Consumes: the existing `_TIER_BY_TOKEN`, `normalize_license`, `extraction_allowed_for` in the same module.
- Produces: `license_token_from_url(raw: str | None) -> str | None` — returns a token in the existing vocabulary (`cc_by`, `cc_by_nc`, `cc_by_nc_nd`, `cc_by_nc_sa`, `cc0`, …) or `None` when the input is not a recognised Creative Commons URL. Callers feed the result to `normalize_license`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/domain/test_licensing.py` **already exists** with `test_normalize_license` and `test_extraction_allowed`. Do NOT overwrite it. **Append** the tests below, and extend its existing import on line 4 to:

```python
from biolit.domain.licensing import (
    extraction_allowed_for,
    license_token_from_url,
    normalize_license,
)
```

Then append (`import pytest`, `LicenseTier`, `extraction_allowed_for` and `normalize_license` are already imported at the top of the file):

```python


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "cc_by"),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "cc_by_nc"),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "cc_by_nc_nd"),
        ("https://creativecommons.org/licenses/by-nc-sa/4.0/", "cc_by_nc_sa"),
        ("https://creativecommons.org/licenses/by-nc/3.0/", "cc_by_nc"),
        ("https://creativecommons.org/licenses/by-nc-nd/3.0/", "cc_by_nc_nd"),
        ("http://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ],
)
def test_every_licence_form_observed_in_the_live_sample_maps_to_its_token(url, expected):
    """One case per form seen in the spec's 41-article live sample, both versions included.

    The version suffix must be stripped: 3.0 and 4.0 of the same licence are the same
    licence for tier purposes, and the live sample contains both.
    """
    assert license_token_from_url(url) == expected


def test_by_nc_nd_is_not_read_as_by_nc():
    """THE SUBSTRING TRAP, and the reason matching is by exact path segment.

    `by-nc` is a proper substring of both `by-nc-nd` and `by-nc-sa`. A shortest-first
    substring match silently mislabels six of the 41 articles in the live sample. That the
    two tokens happen to share a tier today is COINCIDENCE, not safety -- the tier table can
    change, and this function must be correct independently of it.

    Mutation-verified: replacing the segment match with `if "by-nc" in raw` leaves the
    parametrized test above green for by-nc but fails here.
    """
    assert license_token_from_url("https://creativecommons.org/licenses/by-nc-nd/4.0/") != "cc_by_nc"
    assert license_token_from_url("https://creativecommons.org/licenses/by-nc-sa/4.0/") != "cc_by_nc"


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "This file is available for text mining. It may also be used consistent with fair use.",
        "https://www.cochranelibrary.com/cdsr/editorial-policies",
        "https://www.diabetesjournals.org/content/license",
        "Article reuse guidelines: sagepub.com/journals-permissions",
    ],
)
def test_non_creative_commons_terms_yield_no_token_and_are_therefore_refused(raw):
    """Fails CLOSED, which is the whole safety property.

    Four articles in the live sample carry the NIH author-manuscript statement, whose PROSE
    asserts text mining is permitted while carrying no licence identifier. Permitting on
    parsed prose would weaken a deliberate compliance safeguard; it is refused deliberately,
    and this test is where that decision is enforced. The publisher-specific terms
    (Cochrane, Sage, diabetesjournals) are the same shape.
    """
    token = license_token_from_url(raw)
    assert token is None
    _, tier = normalize_license(token)
    assert tier is LicenseTier.unknown
    assert extraction_allowed_for(tier) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/domain/test_licensing.py -v`
Expected: FAIL — `ImportError: cannot import name 'license_token_from_url'`

- [ ] **Step 3: Add a stub so the tests fail at their assertions, not at import**

tdd-guard blocks jumping to full logic while a test still fails at import. Add to `src/biolit/domain/licensing.py`:

```python
def license_token_from_url(raw: str | None) -> str | None:
    raise NotImplementedError
```

Run: `uv run pytest tests/domain/test_licensing.py -v` — expect `NotImplementedError`, not `ImportError`.

- [ ] **Step 4: Implement**

Add `from urllib.parse import urlsplit` to the top of `src/biolit/domain/licensing.py`, then replace the stub:

```python
def license_token_from_url(raw: str | None) -> str | None:
    """Map a Creative Commons licence URL to the canonical token vocabulary above.

    The dead PMC OA service returned tokens like `CC BY`, which `_canonicalize` handled.
    Its replacement returns URLs, so this is the new front half of the same pipeline; the
    tier table and `extraction_allowed_for` are untouched.

    MATCHING IS BY EXACT PATH SEGMENT, never by substring. `by-nc` is a proper substring of
    `by-nc-nd` and `by-nc-sa`, both of which appear in the live sample, so `in` would
    mislabel them. Anything that is not a recognised Creative Commons URL returns None and
    is therefore refused -- prose asserting reuse rights is not a licence identifier.
    """
    if not raw or not raw.strip():
        return None
    parsed = urlsplit(raw.strip())
    if parsed.netloc.lower().removeprefix("www.") != "creativecommons.org":
        return None
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return None
    if segments[0] == "publicdomain" and segments[1] == "zero":
        return "cc0"
    if segments[0] == "licenses":
        return "cc_" + segments[1].replace("-", "_")
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/domain/test_licensing.py -v`
Expected: PASS (all parametrized cases)

- [ ] **Step 6: Run the substring mutation and paste the output**

Save a copy first — `git diff` is vacuous on untracked files:

```bash
cp src/biolit/domain/licensing.py /tmp/licensing.bak
```

Replace the `segments[0] == "licenses"` branch with a substring match:

```python
    if segments[0] == "licenses":
        for code in ("by-nc", "by-nc-nd", "by-nc-sa", "by"):
            if code in parsed.path:
                return "cc_" + code.replace("-", "_")
```

Run: `uv run pytest tests/domain/test_licensing.py -v`
Expected: `test_by_nc_nd_is_not_read_as_by_nc` FAILS; every other test passes. Record the output in the commit message.

Restore and verify:

```bash
cp /tmp/licensing.bak src/biolit/domain/licensing.py
diff /tmp/licensing.bak src/biolit/domain/licensing.py && echo RESTORED
uv run pytest tests/domain/test_licensing.py -q
```

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/biolit/domain/licensing.py tests/domain/test_licensing.py
uv run ruff check . && uv run pyright
git add src/biolit/domain/licensing.py tests/domain/test_licensing.py
git commit -F - <<'EOF'
feat(licensing): map Creative Commons URLs to canonical tokens

The dead PMC OA service returned tokens like "CC BY"; its replacement
returns URLs, so this is the new front half of the same pipeline. The
tier table and extraction_allowed_for are untouched -- this restores a
broken lookup, it does not redefine a tier.

Matching is by exact path segment. `by-nc` is a proper substring of
`by-nc-nd` and `by-nc-sa`, both present in the live sample, so a
substring match mislabels six of 41 articles. That the two tokens share
a tier today is coincidence, not safety.

Mutation-verified by running the weakened case: a substring match leaves
every other test green and fails only the trap test.

Non-CC terms return None and are refused, including the NIH
"available for text mining" statement -- prose asserting reuse rights is
not a licence identifier, and permitting on it would weaken a deliberate
compliance safeguard.
EOF
```

---

### Task 2: Parse licences out of PMC article XML

**Files:**
- Create: `backend/src/biolit/clients/pmc.py`
- Create: `backend/tests/cassettes/pmc_efetch_cc_by.xml`
- Create: `backend/tests/cassettes/pmc_efetch_restricted_stub.xml`
- Test: `backend/tests/clients/test_pmc.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `licences_by_pmcid(root: ET.Element) -> dict[str, str | None]` — maps a **normalized** PMCID (digits only, no `PMC` prefix) to the licence URL, or `None` when the article carries no licence identifier. Also `normalize_pmcid(raw: str | None) -> str | None`.

- [ ] **Step 1: Write the cassettes**

`backend/tests/cassettes/pmc_efetch_cc_by.xml` — an OA article with a licence, plus a second article whose `<license>` carries only an `xlink:href`, to pin the fallback:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<pmc-articleset>
  <article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
           xmlns:xlink="http://www.w3.org/1999/xlink">
    <front>
      <article-meta>
        <article-id pub-id-type="pmid">11111111</article-id>
        <article-id pub-id-type="pmc">8917620</article-id>
        <permissions>
          <copyright-statement>© The Author(s) 2022</copyright-statement>
          <license>
            <ali:license_ref specific-use="textmining" content-type="ccbylicense">https://creativecommons.org/licenses/by/4.0/</ali:license_ref>
            <license-p>Open Access. This article is licensed under a Creative Commons Attribution 4.0 licence.</license-p>
          </license>
        </permissions>
      </article-meta>
    </front>
    <body><p>Full text body.</p></body>
  </article>
  <article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
           xmlns:xlink="http://www.w3.org/1999/xlink">
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">PMC7654321</article-id>
        <permissions>
          <license xlink:href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
            <license-p>This is an open access article under the CC BY-NC-ND licence.</license-p>
          </license>
        </permissions>
      </article-meta>
    </front>
    <body><p>Full text body.</p></body>
  </article>
</pmc-articleset>
```

`backend/tests/cassettes/pmc_efetch_restricted_stub.xml` — the real `PMC1401093` shape: an article element with no `<permissions>` and no `<body>`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<pmc-articleset>
  <article xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
           xmlns:xlink="http://www.w3.org/1999/xlink">
    <front>
      <article-meta>
        <article-id pub-id-type="pmid">22222222</article-id>
        <article-id pub-id-type="pmc">1401093</article-id>
        <title-group><article-title>A restricted article</article-title></title-group>
      </article-meta>
    </front>
  </article>
</pmc-articleset>
```

- [ ] **Step 2: Write the failing tests**

`backend/tests/clients/test_pmc.py`:

```python
from pathlib import Path
from xml.etree import ElementTree as ET

from biolit.clients.pmc import licences_by_pmcid, normalize_pmcid

CASSETTES = Path(__file__).parent.parent / "cassettes"


def _root(name: str) -> ET.Element:
    return ET.fromstring((CASSETTES / name).read_text(encoding="utf-8"))


def test_normalize_pmcid_strips_the_prefix_so_both_sources_key_alike():
    """PubMed's ArticleIdList reports `PMC8917620`; PMC's own article-id reports the bare
    digits `8917620`. Keying the two together without normalising yields an empty join and
    therefore zero permitted papers -- silently, and indistinguishably from 'nothing is
    licensed'. Both directions and the None case are pinned."""
    assert normalize_pmcid("PMC8917620") == "8917620"
    assert normalize_pmcid("8917620") == "8917620"
    assert normalize_pmcid("pmc8917620") == "8917620"
    assert normalize_pmcid(None) is None
    assert normalize_pmcid("") is None


def test_licence_is_read_from_the_ali_license_ref():
    licences = licences_by_pmcid(_root("pmc_efetch_cc_by.xml"))
    assert licences["8917620"] == "https://creativecommons.org/licenses/by/4.0/"


def test_licence_falls_back_to_the_license_href_when_there_is_no_ali_ref():
    """Not every publisher emits ali:license_ref. The fallback is the `<license>` element's
    xlink:href -- still an identifier, never the element's prose."""
    licences = licences_by_pmcid(_root("pmc_efetch_cc_by.xml"))
    assert licences["7654321"] == "https://creativecommons.org/licenses/by-nc-nd/4.0/"


def test_a_restricted_stub_yields_no_licence_rather_than_a_guess():
    """The real PMC1401093 shape -- the article that 404'd throughout Phase 5. It comes back
    as a ~6KB stub with no <permissions> and no <body>. It must map to None, which refuses.

    Classification keys on the permissions block ALONE. <body> presence is corroborating
    evidence that these are different documents, not a criterion: an article could carry a
    permissive licence with no body shipped, and refusing it on that basis would be wrong.
    """
    licences = licences_by_pmcid(_root("pmc_efetch_restricted_stub.xml"))
    assert licences == {"1401093": None}


def test_prose_is_never_returned_as_a_licence():
    """The NIH 'available for text mining' statement lives in <license-p> prose with no
    identifier. Returning that prose would push a non-licence into the token normalizer and
    invite a permissive match on wording. Only identifiers are ever returned."""
    xml = """<pmc-articleset><article
        xmlns:ali="http://www.niso.org/schemas/ali/1.0/"
        xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-id pub-id-type="pmc">3333333</article-id>
        <permissions><license><license-p>This file is available for text mining.</license-p></license></permissions>
      </article-meta></front></article></pmc-articleset>"""
    assert licences_by_pmcid(ET.fromstring(xml)) == {"3333333": None}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/clients/test_pmc.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.clients.pmc'`

- [ ] **Step 4: Implement**

Create `backend/src/biolit/clients/pmc.py`:

```python
"""Pure parsing of PMC article XML. No HTTP — the caller supplies a parsed root.

Replaces the PMC OA Web Service, whose endpoint is dead (404 with no parameters at all,
serving an NCBI error page rather than the service's own XML error document). This reads
the article's own `<permissions>` block instead, which is the PUBLISHER'S licence
statement — so the Phase 1 rule "never infer rights from PMC presence" holds exactly as
written: presence in PMC yields nothing here; only a licence identifier does.
"""

from xml.etree import ElementTree as ET

_ALI = "{http://www.niso.org/schemas/ali/1.0/}"
_XLINK = "{http://www.w3.org/1999/xlink}"


def normalize_pmcid(raw: str | None) -> str | None:
    """Bare digits, no `PMC` prefix, so both id sources key alike.

    PubMed's ArticleIdList reports `PMC8917620`; PMC's own article-id reports `8917620`.
    Joining the two without this yields an empty intersection and therefore zero permitted
    papers — silently, and indistinguishably from "nothing is licensed".
    """
    if not raw or not raw.strip():
        return None
    return raw.strip().upper().removeprefix("PMC") or None


def _licence_of(article: ET.Element) -> str | None:
    """The licence IDENTIFIER, or None. Never the element's prose."""
    permissions = article.find(".//permissions")
    if permissions is None:
        return None
    ref = permissions.find(f".//{_ALI}license_ref")
    if ref is not None and (ref.text or "").strip():
        return (ref.text or "").strip()
    licence = permissions.find(".//license")
    if licence is not None:
        href = licence.get(f"{_XLINK}href")
        if href and href.strip():
            return href.strip()
    return None


def licences_by_pmcid(root: ET.Element) -> dict[str, str | None]:
    """{normalized PMCID: licence identifier or None} for every article in the response.

    An article present with no licence maps to None rather than being omitted, so the
    caller can distinguish "PMC returned it and it carries no licence" (refuse) from
    "PMC did not return it at all" (also refuse, different drop reason).
    """
    out: dict[str, str | None] = {}
    for article in root.findall(".//article"):
        pmcid = None
        for article_id in article.findall(".//article-id"):
            if article_id.get("pub-id-type") == "pmc":
                pmcid = normalize_pmcid(article_id.text)
                break
        if pmcid is None:
            continue
        out[pmcid] = _licence_of(article)
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/clients/test_pmc.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Run the prose mutation and paste the output**

```bash
cp src/biolit/clients/pmc.py /tmp/pmc.bak
```

In `_licence_of`, add a prose fallback before `return None`:

```python
    if licence is not None:
        text = " ".join("".join(licence.itertext()).split())
        if text:
            return text
```

Run: `uv run pytest tests/clients/test_pmc.py -v`
Expected: `test_prose_is_never_returned_as_a_licence` FAILS; the rest pass.

```bash
cp /tmp/pmc.bak src/biolit/clients/pmc.py
diff /tmp/pmc.bak src/biolit/clients/pmc.py && echo RESTORED
```

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/biolit/clients/pmc.py tests/clients/test_pmc.py
uv run ruff check . && uv run pyright && uv run pytest -q
git add src/biolit/clients/pmc.py tests/clients/test_pmc.py tests/cassettes/pmc_efetch_cc_by.xml tests/cassettes/pmc_efetch_restricted_stub.xml
git commit -F - <<'EOF'
feat(pmc): read licences from the article permissions block

Pure parsing, no HTTP. Replaces the PMC OA Web Service, whose endpoint is
dead -- 404 with no parameters at all, serving an NCBI error page rather
than the service's own XML error document.

Reads the publisher's own <permissions> block, so the Phase 1 rule "never
infer rights from PMC presence" holds exactly: presence yields nothing,
only a licence identifier does.

Only identifiers are returned, never prose. The NIH "available for text
mining" statement lives in <license-p> and asserts reuse rights in
wording alone; returning it would push a non-licence into the token
normalizer and invite a permissive match. Mutation-verified: adding a
prose fallback fails only that test.

normalize_pmcid exists because PubMed reports PMC8917620 and PMC reports
8917620. Joining them unnormalised yields an empty intersection and zero
permitted papers -- silently, and indistinguishably from "nothing is
licensed".

The restricted-stub cassette is the real PMC1401093 shape, the article
that 404'd throughout Phase 5.
EOF
```

---

### Task 3: Rewire the client — batched permissions, no dead endpoint

**Files:**
- Modify: `backend/src/biolit/clients/pubmed.py`
- Modify: `backend/tests/clients/test_pubmed.py`
- Create: `backend/tests/cassettes/pmc_efetch_http_error.xml` *(unused body; the route returns 404)*
- Test: `backend/tests/clients/test_pubmed.py`

**Interfaces:**
- Consumes: `licences_by_pmcid`, `normalize_pmcid` (Task 2); `license_token_from_url` (Task 1).
- Produces: `PubMedClient.efetch(pmids: list[str]) -> list[Paper]` — unchanged signature, working licence classification. `_parse_article(article: ET.Element, licences: Mapping[str, str | None]) -> Paper` is now **synchronous and pure**.

**Note on respx routing:** the PMC call now shares `efetch.fcgi` with the PubMed call. Existing tests mock that URL broadly and would capture both. Every route in this file must key on the `db` param: `respx.get(URL, params__contains={"db": "pubmed"})` and `params__contains={"db": "pmc"}`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/clients/test_pubmed.py` (keep existing imports; add `import pytest` if absent):

```python
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@respx.mock
async def test_efetch_classifies_from_the_permissions_block(settings):
    """The replacement path end to end: PubMed article -> its PMC id -> the permissions
    block -> a tier. Routes key on `db` because both calls now share efetch.fcgi."""
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    respx.get(EFETCH, params__contains={"db": "pmc"}).mock(
        return_value=httpx.Response(200, text=_cassette("pmc_efetch_cc_by.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["11111111"])
    paper = papers[0]
    assert paper.license == "cc_by"
    assert paper.license_tier is LicenseTier.open
    assert paper.extraction_allowed is True
    assert paper.text_type is TextType.full_text_unverified


@respx.mock
async def test_a_restricted_stub_is_refused_not_permitted(settings):
    """The PMC1401093 shape. The dangerous failure here is over-permitting, so this pins
    the refusal rather than merely pinning that something was returned."""
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_restricted.xml"))
    )
    respx.get(EFETCH, params__contains={"db": "pmc"}).mock(
        return_value=httpx.Response(200, text=_cassette("pmc_efetch_restricted_stub.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["22222222"])
    paper = papers[0]
    assert paper.license is None
    assert paper.license_tier is LicenseTier.unknown
    assert paper.extraction_allowed is False
    assert paper.text_type is TextType.abstract_only


@respx.mock
async def test_permissions_are_fetched_in_one_batched_call(settings):
    """One request for N papers, not N. The old per-article lookup was the dominant cost of
    a multi-thousand-paper fetch at NCBI's unkeyed 3 req/s, and was the 404's blast radius:
    one article outside the OA subset killed the whole call."""
    pubmed_route = respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    pmc_route = respx.get(EFETCH, params__contains={"db": "pmc"}).mock(
        return_value=httpx.Response(200, text=_cassette("pmc_efetch_cc_by.xml"))
    )
    async with httpx.AsyncClient() as http:
        await PubMedClient(http, settings).efetch(["11111111"])
    assert pubmed_route.call_count == 1
    assert pmc_route.call_count == 1


@respx.mock
async def test_no_pmc_call_is_made_when_no_article_carries_a_pmc_id(settings):
    """respx fails any unmocked request, so leaving the db=pmc route unmocked is what
    proves the call is skipped rather than merely ignored."""
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_no_pmc.xml"))
    )
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["33333333"])
    assert papers[0].text_type is TextType.abstract_only
    assert papers[0].extraction_allowed is False


@respx.mock
async def test_an_http_error_from_the_permissions_call_refuses_rather_than_crashing(settings):
    """THE CASSETTE WHOSE ABSENCE SHIPPED THE DEFECT.

    The suite covered the 200-with-<error> body and had no 404 case, so
    `request_with_retry`'s raise_for_status propagated and one article killed the whole
    fetch -- the same shape as the structured-abstract truncation that survived an
    extensive suite because every cassette happened to hold an unstructured abstract.

    A permissions lookup that fails must degrade to "no licence" (refuse), never to a
    crash and never to a permit.
    """
    respx.get(EFETCH, params__contains={"db": "pubmed"}).mock(
        return_value=httpx.Response(200, text=_cassette("pubmed_efetch_pmc_oa.xml"))
    )
    respx.get(EFETCH, params__contains={"db": "pmc"}).mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as http:
        papers = await PubMedClient(http, settings).efetch(["11111111"])
    assert papers[0].extraction_allowed is False
    assert papers[0].license_tier is LicenseTier.unknown
```

Then **delete** the two now-obsolete tests that mock the dead service — `test_efetch_open_access` and `test_efetch_pmc_but_restricted_license` (replaced by the first two above) — and update `test_efetch_pmc_present_but_not_oa`, `test_efetch_structured_abstract_concatenates_all_sections`, `test_efetch_no_abstract_element_is_none`, and `test_efetch_abstracts_does_not_touch_the_pmc_oa_service` to use `params__contains={"db": "pubmed"}` routing. Rename the last to `test_efetch_abstracts_makes_no_pmc_call`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/clients/test_pubmed.py -v`
Expected: FAIL — the new tests error because `_parse_article` still takes no licences map and still calls the dead endpoint.

- [ ] **Step 3: Implement**

In `src/biolit/clients/pubmed.py`: delete the `_PMC_OA` constant and the whole `_classify_pmc` method. Add imports:

```python
from collections.abc import Mapping

from biolit.clients.pmc import licences_by_pmcid, normalize_pmcid
from biolit.domain.licensing import extraction_allowed_for, license_token_from_url, normalize_license
```

Add a module constant and replace `efetch` / `_parse_article`:

```python
_PMC_ARTICLE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC{pmcid}/"


def _pmc_id_of(article: ET.Element) -> str | None:
    for article_id in article.findall(".//ArticleIdList/ArticleId"):
        if article_id.get("IdType") == "pmc":
            return normalize_pmcid(article_id.text)
    return None


class PubMedClient:
    ...

    async def efetch(self, pmids: list[str]) -> list[Paper]:
        if not pmids:
            return []
        resp = await request_with_retry(
            self._client,
            "GET",
            f"{_EUTILS}/efetch.fcgi",
            retry=self._retry,
            params=self._params(db="pubmed", id=",".join(pmids), retmode="xml"),
        )
        root = ET.fromstring(resp.text)
        articles = root.findall(".//PubmedArticle")
        pmc_ids = sorted({p for article in articles if (p := _pmc_id_of(article))})
        licences = await self._fetch_licences(pmc_ids)
        return [self._parse_article(article, licences) for article in articles]

    async def _fetch_licences(self, pmc_ids: list[str]) -> dict[str, str | None]:
        """One batched request for every PMC id in the response, or none at all.

        Degrades to an empty map on any HTTP failure, which refuses rather than crashing.
        A licence lookup that cannot answer must never be read as permission, and must
        never take the whole fetch down with it -- which is exactly what the dead OA
        endpoint did.
        """
        if not pmc_ids:
            return {}
        try:
            resp = await request_with_retry(
                self._client,
                "GET",
                f"{_EUTILS}/efetch.fcgi",
                retry=self._retry,
                params=self._params(db="pmc", id=",".join(pmc_ids), retmode="xml"),
            )
        except httpx.HTTPError:
            return {}
        try:
            return licences_by_pmcid(ET.fromstring(resp.text))
        except ET.ParseError:
            return {}

    def _parse_article(self, article: ET.Element, licences: Mapping[str, str | None]) -> Paper:
        ...
        # unchanged parsing of pmid/title/abstract/journal/year/authors/mesh/doi
        pmc_id = _pmc_id_of(article)
        raw_licence = licences.get(pmc_id) if pmc_id else None
        token, tier = normalize_license(license_token_from_url(raw_licence))
        if raw_licence:
            text_type = TextType.full_text_unverified
            pointer = _PMC_ARTICLE_URL.format(pmcid=pmc_id)
        else:
            text_type = TextType.abstract_only
            pointer = None
        return Paper(
            ...,
            text_type=text_type,
            full_text_pointer=pointer,
            license=token,
            license_tier=tier,
            extraction_allowed=extraction_allowed_for(tier),
            raw={"pmid": pmid, "pmc_id": pmc_id},
        )
```

Keep every other line of `_parse_article` as it is; only the trailing classification block and the signature change. Note `_parse_article` is no longer `async` — remove the keyword and the `await` at its call site.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/clients/ -v`
Expected: PASS

- [ ] **Step 5: Run the over-permit mutation and paste the output**

```bash
cp src/biolit/clients/pubmed.py /tmp/pubmed.bak
```

Make a failed lookup permissive — change `_fetch_licences`'s `except httpx.HTTPError: return {}` to re-raise, and separately try treating PMC presence as permission by replacing the classification block with `tier = LicenseTier.open if pmc_id else LicenseTier.unknown`.

Run: `uv run pytest tests/clients/test_pubmed.py -v` after each.
Expected: the re-raise fails `test_an_http_error_from_the_permissions_call_refuses_rather_than_crashing`; the presence-implies-permission change fails `test_a_restricted_stub_is_refused_not_permitted`.

```bash
cp /tmp/pubmed.bak src/biolit/clients/pubmed.py
diff /tmp/pubmed.bak src/biolit/clients/pubmed.py && echo RESTORED
```

- [ ] **Step 6: Verify against the live service, once**

This is the only network call in the plan and it is free. It confirms the fix works against reality rather than only against cassettes:

```bash
uv run python -c "
import asyncio, httpx
from biolit.clients.pubmed import PubMedClient
from biolit.config import get_settings
async def main():
    async with httpx.AsyncClient(timeout=60) as h:
        c = PubMedClient(h, get_settings())
        pmids = await c.esearch('metformin cardiovascular outcomes', retmax=20)
        papers = await c.efetch(pmids)
    ok = sum(1 for p in papers if p.extraction_allowed)
    print(f'{ok}/{len(papers)} extraction_allowed')
    print(sorted({str(p.license) for p in papers}))
asyncio.run(main())
"
```

Expected: no exception, and a **non-zero** allowed count (the spec measured ~53% on comparable queries). If it raises or reports 0/20, stop and report — the fix has not worked and the remaining tasks are built on it.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/biolit/clients/pubmed.py tests/clients/test_pubmed.py
uv run ruff check . && uv run pyright && uv run pytest -q
git add src/biolit/clients/pubmed.py tests/clients/test_pubmed.py
git commit -F - <<'EOF'
fix(pubmed): classify licences from PMC permissions, not the dead OA service

The configured PMC OA endpoint returns 404 with no parameters at all,
serving an NCBI error page. Measured consequence before this change:
0/20 papers extractable on a plain query AND 0/20 restricted to the PMC
open-access subset, with every paper carrying an abstract. The chain was
deterministic -- no licence, unknown tier, extraction_allowed False,
build_record returns None for every paper.

The known efetch 404 defect had that dead endpoint as its cause rather
than merely its trigger, so this is one fix and not two.

Permissions now come from efetch db=pmc in ONE batched call for all N
papers instead of N sequential per-article lookups. That also makes
_parse_article synchronous and pure, so it is testable with no network at
all -- previously every test of it had to mock a per-article HTTP call.

Both routes now share efetch.fcgi, so every respx route keys on the db
param.

Adds the two cassettes whose absence let this ship: an HTTP 404 on the
permissions call, and a permissions-less restricted stub. A failed lookup
degrades to "no licence" -- refusing, never crashing and never permitting.
Mutation-verified in both directions: re-raising the HTTP error and
treating PMC presence as permission each fail exactly one test.
EOF
```

---

### Task 4: `StageStatus`, `StageReport`, and `PipelineState.stages`

**Files:**
- Modify: `backend/src/biolit/state/pipeline.py`
- Test: `backend/tests/state/test_pipeline.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `StageStatus` (StrEnum: `completed`, `not_implemented`), `StageReport(name: str, status: StageStatus, n_in: int, n_out: int, dropped: dict[str, int], note: str | None)`, and `PipelineState.stages: list[StageReport]`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/state/` exists and contains `__init__.py` and `test_adapters.py`; **create** `backend/tests/state/test_pipeline.py`:

```python
import json

from biolit.state.pipeline import PipelineState, StageReport, StageStatus


def test_a_fresh_state_has_no_stages_and_keeps_its_existing_defaults():
    state = PipelineState(question="does metformin cause lactic acidosis?")
    assert state.stages == []
    assert state.contradictions == []
    assert state.answer is None


def test_not_implemented_survives_the_json_dump():
    """THE WHOLE REASON THIS FIELD EXISTS.

    `contradictions: []` is indistinguishable from "the Critic ran and found nothing". The
    status lives in PipelineState rather than only in the printed report specifically so a
    MACHINE reader of the JSON dump also sees not_implemented, and never reads an empty
    list as a negative result.
    """
    state = PipelineState(
        question="q",
        stages=[
            StageReport(
                name="critic",
                status=StageStatus.not_implemented,
                n_in=3,
                n_out=0,
                note="see ADR-0017",
            )
        ],
    )
    payload = json.loads(state.model_dump_json())
    assert payload["contradictions"] == []
    assert payload["stages"][0]["status"] == "not_implemented"
    assert "ADR-0017" in payload["stages"][0]["note"]


def test_stage_status_is_a_strenum_so_it_serialises_as_its_value():
    """ADR-0005. A plain Enum serialises as `StageStatus.completed`, which is not what a
    JSON consumer can match on."""
    assert StageStatus.completed == "completed"
    assert json.dumps({"s": StageStatus.completed}) == '{"s": "completed"}'


def test_dropped_defaults_to_an_empty_dict_and_is_not_shared_between_reports():
    """A mutable default shared across instances is the classic pydantic-adjacent bug; the
    ledger would then accumulate another stage's drops."""
    first = StageReport(name="a", status=StageStatus.completed, n_in=1, n_out=1)
    second = StageReport(name="b", status=StageStatus.completed, n_in=1, n_out=1)
    first.dropped["x"] = 1
    assert second.dropped == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/state/test_pipeline.py -v`
Expected: FAIL — `ImportError: cannot import name 'StageReport'`

- [ ] **Step 3: Implement**

Replace `backend/src/biolit/state/pipeline.py`'s contents, keeping the existing eight fields exactly:

```python
from enum import StrEnum

from pydantic import BaseModel, Field

from biolit.domain.paper import Paper
from biolit.domain.records import Citation, Cluster, ContradictionFinding, ExtractedRecord


class StageStatus(StrEnum):
    """Only the two states something actually produces today.

    No `blocked`/`skipped`/`failed` member exists until a stage emits one — ADR-0013's
    "no infrastructure without a demonstrated consumer".
    """

    completed = "completed"
    not_implemented = "not_implemented"


class StageReport(BaseModel):
    """One stage's accounting, including what it dropped and why.

    Lives in PipelineState rather than only in the printed report so the JSON dump carries
    it too: `contradictions: []` alone is indistinguishable from "ran and found nothing",
    and a machine reader must be able to tell those apart.
    """

    name: str
    status: StageStatus
    n_in: int
    n_out: int
    dropped: dict[str, int] = Field(default_factory=dict)
    note: str | None = None


class PipelineState(BaseModel):
    question: str
    sub_queries: list[str] = Field(default_factory=list)
    candidate_papers: list[Paper] = Field(default_factory=list)
    extracted_records: dict[str, ExtractedRecord] = Field(default_factory=dict)
    clusters: list[Cluster] = Field(default_factory=list)
    contradictions: list[ContradictionFinding] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    answer: str | None = None
    stages: list[StageReport] = Field(default_factory=list)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/state/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/biolit/state/pipeline.py tests/state/test_pipeline.py
uv run ruff check . && uv run pyright && uv run pytest -q
git add src/biolit/state/pipeline.py tests/state/test_pipeline.py
git commit -F - <<'EOF'
feat(state): add per-stage reporting to PipelineState

One new field. The existing eight are unchanged -- nothing about the
happy path needed a new one; what was missing was any notion of stage
STATUS.

`contradictions: []` is indistinguishable from "the Critic ran and found
nothing". Putting the status in PipelineState rather than only in the
printed report means the JSON dump carries it too, so a machine reader
sees not_implemented instead of reading an empty list as a negative
result.

StageStatus has exactly the two members something produces today. No
blocked/skipped/failed until a stage emits one, per ADR-0013.
EOF
```

---

### Task 5: Entity and record stages, with the drop ledger

**Files:**
- Create: `backend/src/biolit/pipeline/__init__.py` *(empty)*
- Create: `backend/src/biolit/pipeline/stages.py`
- Test: `backend/tests/pipeline/__init__.py` *(empty)*, `backend/tests/pipeline/test_stages.py`

**Interfaces:**
- Consumes: `StageReport`, `StageStatus` (Task 4); `build_record`, `SameSentenceAsEntitiesExtractor`, `canonicalize`, `extract_entities` from existing modules.
- Produces:
  - `RETRIEVE`, `NER_LINKING`, `LICENCE_GATE`, `EXTRACT`, `CLUSTER`, `CRITIC`, `SYNTHESIS` — stage-name constants.
  - `retrieve_stage(pmids: Sequence[str], papers: Sequence[Paper]) -> StageReport`
  - `entities_stage(papers, *, extract, linker) -> tuple[dict[str, list[Entity]], StageReport]` where `extract` is `Callable[[str], list[Entity]]`
  - `RecordOutcome(records: dict[str, ExtractedRecord], licence: StageReport, extract: StageReport)`
  - `records_stage(papers, entities_by_paper) -> RecordOutcome`

- [ ] **Step 1: Write the failing tests**

`backend/tests/pipeline/test_stages.py`:

```python
from biolit.domain.enums import EntityLabel, LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.pipeline.stages import (
    entities_stage,
    records_stage,
    retrieve_stage,
)
from biolit.state.pipeline import StageStatus


def _paper(pid: str, abstract: str | None, *, allowed: bool) -> Paper:
    return Paper(
        id=pid,
        source=Source.pubmed,
        pmid=pid,
        title=f"Title {pid}",
        abstract=abstract,
        text_type=TextType.abstract_only,
        license_tier=LicenseTier.open if allowed else LicenseTier.unknown,
        license="cc_by" if allowed else None,
        extraction_allowed=allowed,
    )


def test_retrieve_stage_counts_pmids_that_produced_no_paper_and_papers_with_no_abstract():
    papers = [_paper("a", "Metformin causes acidosis.", allowed=True), _paper("b", None, allowed=True)]
    report = retrieve_stage(["1", "2", "3"], papers)
    assert report.status is StageStatus.completed
    assert report.n_in == 3
    assert report.n_out == 2
    assert report.dropped["no_abstract"] == 1


def test_entities_stage_counts_unlinked_entities_without_discarding_them():
    """NIL entities are COUNTED, not dropped. `SameSentenceAsEntitiesExtractor` already
    fails closed on an unlinked entity; removing them here would double-count the same loss
    and make the ledger disagree with what the extractor actually saw."""
    linked = Entity(text="metformin", label=EntityLabel.CHEMICAL, start=0, end=9,
                    canonical_id="D008687", canonical_name="Metformin")
    nil = Entity(text="wibble", label=EntityLabel.DISEASE, start=10, end=16)

    entities, report = entities_stage(
        [_paper("a", "metformin wibble", allowed=True)],
        extract=lambda text: [linked, nil],
        linker=None,
    )
    assert entities["a"] == [linked, nil]
    assert report.dropped["entity_unlinked"] == 1
    assert report.n_out == 2


def test_records_stage_reports_the_licence_refusal_broken_out_by_licence():
    """The licence gate is `build_record`, which returns None for a refused paper. The
    ledger must break refusals out by the licence actually seen, because on real data
    roughly half of retrieved papers are correctly refused and a bare count reads as a bug.
    """
    papers = [
        _paper("ok", "Metformin causes acidosis.", allowed=True),
        _paper("no", "Metformin causes acidosis.", allowed=False),
    ]
    outcome = records_stage(papers, {"ok": [], "no": []})
    assert set(outcome.records) == {"ok"}
    assert outcome.licence.n_in == 2
    assert outcome.licence.n_out == 1
    assert outcome.licence.dropped == {"licence_refused:none": 1}


def test_records_stage_counts_records_that_yielded_no_findings():
    """A permitted paper with no chemical+disease sentence produces an empty record. That
    is a real outcome, not an error -- but it must be visible, or an empty final result
    looks like a crash."""
    paper = _paper("ok", "This abstract mentions nothing linkable.", allowed=True)
    outcome = records_stage([paper], {"ok": []})
    assert outcome.extract.dropped["zero_findings"] == 1
    assert outcome.extract.n_out == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_stages.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.pipeline'`

- [ ] **Step 3: Implement**

Create `backend/src/biolit/pipeline/__init__.py` (empty) and `backend/src/biolit/pipeline/stages.py`:

```python
"""Pure stage functions for the end-to-end pipeline.

Each returns its result plus a `StageReport`. Nothing here touches the network or loads a
model — `__main__` injects both — so every stage is testable offline.

THE DROP LEDGER IS THE POINT. On real data most of the interesting behaviour is drops:
roughly half of retrieved papers are correctly refused by the licence gate, and a bare
count of that reads as a bug on first run. Every stage records what it dropped and why.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from biolit.domain.paper import Paper
from biolit.domain.records import Entity, ExtractedRecord
from biolit.extract.base import build_record
from biolit.extract.deterministic import SameSentenceAsEntitiesExtractor
from biolit.state.pipeline import StageReport, StageStatus

RETRIEVE = "retrieve"
NER_LINKING = "ner_linking"
LICENCE_GATE = "licence_gate"
EXTRACT = "extract"
CLUSTER = "cluster"
CRITIC = "critic"
SYNTHESIS = "synthesis"


def retrieve_stage(pmids: Sequence[str], papers: Sequence[Paper]) -> StageReport:
    dropped: dict[str, int] = {}
    no_abstract = sum(1 for paper in papers if not paper.abstract)
    if no_abstract:
        dropped["no_abstract"] = no_abstract
    unparsed = len(pmids) - len(papers)
    if unparsed > 0:
        dropped["no_article_returned"] = unparsed
    return StageReport(
        name=RETRIEVE,
        status=StageStatus.completed,
        n_in=len(pmids),
        n_out=len(papers),
        dropped=dropped,
    )


def entities_stage(
    papers: Sequence[Paper],
    *,
    extract: Callable[[str], list[Entity]],
    linker: object,
) -> tuple[dict[str, list[Entity]], StageReport]:
    """Run NER + linking over each paper's abstract.

    `extract` is injected already bound to its model and linker so this function stays
    free of heavy imports. Unlinked entities are COUNTED, never removed: the extractor
    already fails closed on them, and dropping them here would double-count one loss.
    """
    by_paper: dict[str, list[Entity]] = {}
    total = 0
    unlinked = 0
    for paper in papers:
        entities = extract(paper.abstract or "")
        by_paper[paper.id] = entities
        total += len(entities)
        unlinked += sum(1 for entity in entities if entity.canonical_id is None)
    dropped = {"entity_unlinked": unlinked} if unlinked else {}
    return by_paper, StageReport(
        name=NER_LINKING,
        status=StageStatus.completed,
        n_in=len(papers),
        n_out=total,
        dropped=dropped,
    )


@dataclass(frozen=True)
class RecordOutcome:
    records: dict[str, ExtractedRecord]
    licence: StageReport
    extract: StageReport


def records_stage(
    papers: Sequence[Paper], entities_by_paper: Mapping[str, Sequence[Entity]]
) -> RecordOutcome:
    """Apply the licence gate and build records, reporting both stages separately.

    `build_record` IS the licence gate and the single enforcement point — it returns None
    for a paper whose licence forbids extraction, and suppresses the whole record rather
    than just the findings, because Entity.text and Finding.text both carry verbatim
    abstract substrings. This function must never pre-filter and call the extractor
    directly; that would move the gate.
    """
    extractor = SameSentenceAsEntitiesExtractor(entities_by_paper)
    records: dict[str, ExtractedRecord] = {}
    refused: dict[str, int] = {}
    zero_findings = 0
    for paper in papers:
        record = build_record(
            paper, entities=list(entities_by_paper.get(paper.id, ())), extractor=extractor
        )
        if record is None:
            key = f"licence_refused:{paper.license or 'none'}"
            refused[key] = refused.get(key, 0) + 1
            continue
        records[paper.id] = record
        if not record.key_findings:
            zero_findings += 1
    return RecordOutcome(
        records=records,
        licence=StageReport(
            name=LICENCE_GATE,
            status=StageStatus.completed,
            n_in=len(papers),
            n_out=len(records),
            dropped=refused,
            note=(
                "Refused papers carry no Creative Commons licence in the publisher's "
                "permissions block (LicenseTier.unknown). This is the Phase 1 compliance "
                "rule working as designed, not a failure."
            ),
        ),
        extract=StageReport(
            name=EXTRACT,
            status=StageStatus.completed,
            n_in=len(records),
            n_out=len(records),
            dropped={"zero_findings": zero_findings} if zero_findings else {},
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/test_stages.py -v`
Expected: PASS

- [ ] **Step 5: Run the licence-gate-bypass mutation and paste the output**

```bash
cp src/biolit/pipeline/stages.py /tmp/stages.bak
```

Replace the `build_record` call with a direct extractor call that skips the gate:

```python
        record = ExtractedRecord(
            paper_id=paper.id,
            entities=list(entities_by_paper.get(paper.id, ())),
            key_findings=extractor.findings(paper),
        )
```

Run: `uv run pytest tests/pipeline/test_stages.py -v`
Expected: `test_records_stage_reports_the_licence_refusal_broken_out_by_licence` FAILS — the refused paper is now extracted.

```bash
cp /tmp/stages.bak src/biolit/pipeline/stages.py
diff /tmp/stages.bak src/biolit/pipeline/stages.py && echo RESTORED
```

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/biolit/pipeline/ tests/pipeline/
uv run ruff check . && uv run pyright && uv run pytest -q
git add src/biolit/pipeline/ tests/pipeline/
git commit -F - <<'EOF'
feat(pipeline): entity and record stages with the drop ledger

Pure functions, no network and no model loading -- __main__ injects both.

The ledger is the point. On real data roughly half of retrieved papers
are correctly refused by the licence gate, so refusals are broken out by
the licence actually seen and carry a note saying the rule is working as
designed. A bare count reads as a bug on first run.

build_record IS the licence gate and the single enforcement point; it
suppresses the whole record rather than just the findings, because
Entity.text and Finding.text both carry verbatim abstract substrings.
records_stage calls it rather than pre-filtering and invoking the
extractor directly, which would move the gate. Mutation-verified:
bypassing build_record fails exactly the refusal test.

Unlinked entities are counted, never removed -- the extractor already
fails closed on them, so dropping them here would double-count one loss
and make the ledger disagree with what the extractor saw.
EOF
```

---

### Task 6: Cluster stage and the two unimplemented stubs

**Files:**
- Modify: `backend/src/biolit/pipeline/stages.py`
- Test: `backend/tests/pipeline/test_stages.py`

**Interfaces:**
- Consumes: `cluster_papers`, `SameSentencePairing` from `biolit.cluster`; `StageReport`/`StageStatus` (Task 4).
- Produces:
  - `cluster_stage(records, *, texts, pairing, min_size=2) -> tuple[list[Cluster], StageReport]`
  - `critic_stub(n_clusters: int) -> StageReport`
  - `synthesis_stub() -> StageReport`
  - `ADR_0017_NOTE: str`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/pipeline/test_stages.py` (extend the import from `biolit.pipeline.stages` to include `cluster_stage`, `critic_stub`, `synthesis_stub`):

```python
from biolit.cluster.pairing import SameSentencePairing
from biolit.domain.records import ExtractedRecord


def test_cluster_stage_separates_singleton_keys_from_papers_that_produced_no_pair():
    """Two different drops with the same visible effect (no cluster) and different causes.
    Collapsing them would hide which one is happening on a small result set."""
    shared = "Metformin causes acidosis."
    entities = [
        Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9,
               canonical_id="D008687", canonical_name="Metformin"),
        Entity(text="acidosis", label=EntityLabel.DISEASE, start=17, end=25,
               canonical_id="D000138", canonical_name="Acidosis"),
    ]
    records = [
        ExtractedRecord(paper_id="a", entities=entities),
        ExtractedRecord(paper_id="b", entities=entities),
        ExtractedRecord(paper_id="c", entities=[]),
    ]
    texts = {"a": shared, "b": shared, "c": "Nothing here."}

    clusters, report = cluster_stage(records, texts=texts, pairing=SameSentencePairing())

    assert [c.paper_ids for c in clusters] == [["a", "b"]]
    assert report.dropped.get("no_pairs") == 1
    assert report.n_out == 1


def test_cluster_stage_counts_a_key_held_by_only_one_paper_as_a_singleton_drop():
    entities_a = [
        Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9,
               canonical_id="D008687", canonical_name="Metformin"),
        Entity(text="acidosis", label=EntityLabel.DISEASE, start=17, end=25,
               canonical_id="D000138", canonical_name="Acidosis"),
    ]
    records = [ExtractedRecord(paper_id="a", entities=entities_a)]
    clusters, report = cluster_stage(
        records, texts={"a": "Metformin causes acidosis."}, pairing=SameSentencePairing()
    )
    assert clusters == []
    assert report.dropped["singleton_key"] == 1


def test_critic_stub_reports_not_implemented_and_points_at_the_adr():
    """Not a placeholder result. A stub that returned any ContradictionFinding at all would
    be worse than an empty list, because it manufactures a result where none exists."""
    report = critic_stub(n_clusters=4)
    assert report.status is StageStatus.not_implemented
    assert report.n_in == 4
    assert report.n_out == 0
    assert "ADR-0017" in (report.note or "")


def test_synthesis_stub_says_not_yet_built_rather_than_citing_a_decision():
    """Synthesis is unimplemented for a DIFFERENT reason than the Critic: no ADR retired
    it, it was simply never built. Dressing that up as a decision would misrepresent the
    project's own record."""
    report = synthesis_stub()
    assert report.status is StageStatus.not_implemented
    assert "ADR" not in (report.note or "")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/test_stages.py -v`
Expected: FAIL — `ImportError: cannot import name 'cluster_stage'`

- [ ] **Step 3: Implement**

Append to `backend/src/biolit/pipeline/stages.py` (add `from biolit.cluster.group import cluster_papers`, `from biolit.cluster.pairing import PairingStrategy`, `from biolit.domain.records import Cluster` to its imports):

```python
ADR_0017_NOTE = (
    "Contradiction detection is not implemented. The CTD-derived gold standard was "
    "retired by Gate 2 (pi-hat = 0.067) and the re-scoped alternative was declined; "
    "see ADR-0017."
)


def cluster_stage(
    records: Sequence[ExtractedRecord],
    *,
    texts: Mapping[str, str],
    pairing: PairingStrategy,
    min_size: int = 2,
) -> tuple[list[Cluster], StageReport]:
    """Group papers by shared chemical|disease key, per ADR-0013's SameSentencePairing.

    Calls `cluster_papers` twice rather than reimplementing its grouping: once at
    min_size=1 to see every key, then filters. That keeps the validated function as the
    single source of grouping logic while still exposing WHICH drop occurred — a singleton
    key and a paper that produced no pair at all look identical in the output and have
    entirely different causes.
    """
    all_keys = cluster_papers(records, texts=texts, pairing=pairing, min_size=1)
    clusters = [cluster for cluster in all_keys if len(cluster.paper_ids) >= min_size]

    dropped: dict[str, int] = {}
    singletons = len(all_keys) - len(clusters)
    if singletons:
        dropped["singleton_key"] = singletons
    paired = {paper_id for cluster in all_keys for paper_id in cluster.paper_ids}
    no_pairs = sum(1 for record in records if record.paper_id not in paired)
    if no_pairs:
        dropped["no_pairs"] = no_pairs

    return clusters, StageReport(
        name=CLUSTER,
        status=StageStatus.completed,
        n_in=len(records),
        n_out=len(clusters),
        dropped=dropped,
    )


def critic_stub(n_clusters: int) -> StageReport:
    """Reports that the Critic does not exist. Emits no ContradictionFinding, ever."""
    return StageReport(
        name=CRITIC,
        status=StageStatus.not_implemented,
        n_in=n_clusters,
        n_out=0,
        note=ADR_0017_NOTE,
    )


def synthesis_stub() -> StageReport:
    """Unimplemented for a different reason than the Critic: never built, not retired."""
    return StageReport(
        name=SYNTHESIS,
        status=StageStatus.not_implemented,
        n_in=0,
        n_out=0,
        note="Answer synthesis and citation assembly are not yet built.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/biolit/pipeline/ tests/pipeline/
uv run ruff check . && uv run pyright && uv run pytest -q
git add src/biolit/pipeline/stages.py tests/pipeline/test_stages.py
git commit -F - <<'EOF'
feat(pipeline): cluster stage and the two unimplemented stubs

cluster_stage calls cluster_papers twice -- once at min_size=1 to see
every key, then filters -- rather than reimplementing the grouping. That
keeps ADR-0013's validated function as the single source of that logic
while still distinguishing a singleton key from a paper that produced no
pair at all. Both look identical in the output and have entirely
different causes.

Two stubs, not one. The Critic was the stage named in the request, but
answer and citations belong to a Synthesis stage that was never built
either, and marking only the Critic would move the silent gap one field
over rather than close it. Their notes differ because their reasons do:
the Critic points at ADR-0017 as deliberately retired, Synthesis says
plainly that it does not exist yet and cites no decision, because none
was made.

Neither emits a placeholder value. A stub returning any finding at all
would be worse than an empty list -- it manufactures a result where none
exists.
EOF
```

---

### Task 7: Report rendering, the CLI, and the end-to-end test

**Files:**
- Create: `backend/src/biolit/pipeline/report.py`
- Create: `backend/src/biolit/pipeline/__main__.py`
- Test: `backend/tests/pipeline/test_report.py`, `backend/tests/pipeline/test_end_to_end_wiring.py`

**Interfaces:**
- Consumes: everything from Tasks 4–6.
- Produces: `render_report(stages: Sequence[StageReport]) -> str`; `main(argv: list[str] | None = None) -> None`.

- [ ] **Step 1: Write the failing tests**

`backend/tests/pipeline/test_report.py`:

```python
from biolit.pipeline.report import render_report
from biolit.pipeline.stages import ADR_0017_NOTE
from biolit.state.pipeline import StageReport, StageStatus


def test_a_not_implemented_stage_is_labelled_and_carries_its_note():
    """A user running this must see "not implemented", never a silent gap. The note is
    rendered too, so the reason travels with the label."""
    text = render_report(
        [StageReport(name="critic", status=StageStatus.not_implemented, n_in=3, n_out=0,
                     note=ADR_0017_NOTE)]
    )
    assert "NOT IMPLEMENTED" in text
    assert "ADR-0017" in text


def test_licence_refusals_are_rendered_with_the_rule_that_caused_them():
    """With roughly half of real papers correctly refused, a bare count reads as a bug.
    The rendered block has to make the refusal legible as intended behaviour."""
    text = render_report(
        [StageReport(name="licence_gate", status=StageStatus.completed, n_in=60, n_out=32,
                     dropped={"licence_refused:none": 28},
                     note="This is the Phase 1 compliance rule working as designed, not a failure.")]
    )
    assert "32" in text and "28" in text
    assert "working as designed" in text


def test_a_stage_with_no_drops_renders_without_an_empty_drop_block():
    text = render_report(
        [StageReport(name="retrieve", status=StageStatus.completed, n_in=5, n_out=5)]
    )
    assert "dropped" not in text.lower()
```

`backend/tests/pipeline/test_end_to_end_wiring.py`:

```python
from biolit.cluster.pairing import SameSentencePairing
from biolit.domain.enums import EntityLabel, LicenseTier, Source, TextType
from biolit.domain.paper import Paper
from biolit.domain.records import Entity
from biolit.pipeline.stages import (
    cluster_stage,
    critic_stub,
    entities_stage,
    records_stage,
    retrieve_stage,
    synthesis_stub,
)
from biolit.state.pipeline import PipelineState, StageStatus

ABSTRACT = "Metformin therapy was associated with acidosis in this cohort."


def _entities() -> list[Entity]:
    return [
        Entity(text="Metformin", label=EntityLabel.CHEMICAL, start=0, end=9,
               canonical_id="D008687", canonical_name="Metformin"),
        Entity(text="acidosis", label=EntityLabel.DISEASE, start=38, end=46,
               canonical_id="D000138", canonical_name="Acidosis"),
    ]


def _paper(pid: str, *, allowed: bool) -> Paper:
    return Paper(
        id=pid, source=Source.pubmed, pmid=pid, title=f"Title {pid}", abstract=ABSTRACT,
        text_type=TextType.abstract_only,
        license_tier=LicenseTier.open if allowed else LicenseTier.unknown,
        license="cc_by" if allowed else None,
        extraction_allowed=allowed,
    )


def test_the_whole_pipeline_runs_and_the_ledger_accounts_for_every_paper():
    """The wiring test. Two permitted papers cluster; one refused paper is accounted for.

    Asserts the LEDGER, not just the outputs -- the ledger is the deliverable, and it is
    the thing most likely to regress silently.
    """
    papers = [_paper("a", allowed=True), _paper("b", allowed=True), _paper("c", allowed=False)]
    state = PipelineState(question="does metformin cause acidosis?")
    state.candidate_papers = papers
    state.stages.append(retrieve_stage(["1", "2", "3"], papers))

    entities, entity_report = entities_stage(
        papers, extract=lambda text: _entities(), linker=None
    )
    state.stages.append(entity_report)

    outcome = records_stage(papers, entities)
    state.extracted_records = outcome.records
    state.stages.extend([outcome.licence, outcome.extract])

    clusters, cluster_report = cluster_stage(
        list(outcome.records.values()),
        texts={p.id: ABSTRACT for p in papers},
        pairing=SameSentencePairing(),
    )
    state.clusters = clusters
    state.stages.append(cluster_report)
    state.stages.append(critic_stub(len(clusters)))
    state.stages.append(synthesis_stub())

    assert set(state.extracted_records) == {"a", "b"}
    assert [c.paper_ids for c in state.clusters] == [["a", "b"]]

    by_name = {stage.name: stage for stage in state.stages}
    assert by_name["licence_gate"].dropped == {"licence_refused:none": 1}
    assert by_name["critic"].status is StageStatus.not_implemented
    assert by_name["synthesis"].status is StageStatus.not_implemented


def test_both_unimplemented_stages_survive_the_json_dump():
    """The requirement most likely to regress silently, so it gets its own test: a machine
    reader of the dump must see not_implemented rather than an empty contradictions list."""
    state = PipelineState(question="q", stages=[critic_stub(0), synthesis_stub()])
    payload = state.model_dump(mode="json")
    statuses = {stage["name"]: stage["status"] for stage in payload["stages"]}
    assert statuses == {"critic": "not_implemented", "synthesis": "not_implemented"}
    assert payload["contradictions"] == []
    assert payload["answer"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/pipeline/ -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'biolit.pipeline.report'`

- [ ] **Step 3: Implement `report.py`**

```python
"""Render the stage ledger as the human-readable block the CLI prints."""

from collections.abc import Sequence

from biolit.state.pipeline import StageReport, StageStatus


def _render_one(stage: StageReport) -> list[str]:
    if stage.status is StageStatus.not_implemented:
        lines = [f"{stage.name:<14} NOT IMPLEMENTED"]
        if stage.note:
            lines.append(f"{'':<14}   {stage.note}")
        return lines

    lines = [f"{stage.name:<14} {stage.n_in} in -> {stage.n_out} out"]
    if stage.note:
        lines.append(f"{'':<14}   {stage.note}")
    for reason, count in sorted(stage.dropped.items()):
        lines.append(f"{'':<14}   dropped {count}: {reason}")
    return lines


def render_report(stages: Sequence[StageReport]) -> str:
    out: list[str] = []
    for stage in stages:
        out.extend(_render_one(stage))
    return "\n".join(out)
```

- [ ] **Step 4: Implement `__main__.py`**

```python
"""CLI: run a query through the real Phase 1-5 components and print the stage ledger.

No LLM calls and no paid API calls. Every component is either local (the NER checkpoint,
the MeSH dictionary) or a free NCBI endpoint, so there is no pricing step and no
authorization gate -- there is nothing to authorize.

`main()` gets no direct unit test per project convention; every function it calls is
tested in its own module.
"""


def main(argv: list[str] | None = None) -> None:
    # Heavy imports local to main, same pattern as end_to_end.main (E402 exemption).
    import argparse
    import asyncio
    from pathlib import Path

    import httpx

    from biolit.canon.canonicalize import canonicalize
    from biolit.canon.linker import DictionaryLinker
    from biolit.canon.mesh import MeshDictionary
    from biolit.clients.pubmed import PubMedClient
    from biolit.cluster.pairing import SameSentencePairing
    from biolit.config import get_settings
    from biolit.ner.extract import extract_entities
    from biolit.ner.model import NerModel
    from biolit.pipeline.report import render_report
    from biolit.pipeline.stages import (
        cluster_stage,
        critic_stub,
        entities_stage,
        records_stage,
        retrieve_stage,
        synthesis_stub,
    )
    from biolit.state.pipeline import PipelineState

    parser = argparse.ArgumentParser(description="Run a query through the BioLit pipeline.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--max-papers", type=int, default=20)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()

    async def retrieve():
        async with httpx.AsyncClient(timeout=60) as http:
            client = PubMedClient(http, settings)
            pmids = await client.esearch(args.query, retmax=args.max_papers)
            return pmids, await client.efetch(pmids)

    pmids, papers = asyncio.run(retrieve())

    state = PipelineState(question=args.query)
    state.candidate_papers = papers
    state.stages.append(retrieve_stage(pmids, papers))

    model = NerModel.load(settings)
    linker = DictionaryLinker(MeshDictionary.from_artifact(settings.mesh_artifact_path))

    def extract(text: str):
        entities = extract_entities(text, model, score_threshold=settings.ner_score_threshold)
        return canonicalize(entities, text, linker=linker)

    entities_by_paper, entity_report = entities_stage(papers, extract=extract, linker=linker)
    state.stages.append(entity_report)

    outcome = records_stage(papers, entities_by_paper)
    state.extracted_records = outcome.records
    state.stages.extend([outcome.licence, outcome.extract])

    texts = {paper.id: paper.abstract or "" for paper in papers}
    clusters, cluster_report = cluster_stage(
        list(outcome.records.values()), texts=texts, pairing=SameSentencePairing()
    )
    state.clusters = clusters
    state.stages.append(cluster_report)

    state.stages.append(critic_stub(len(clusters)))
    state.stages.append(synthesis_stub())

    print(f"query: {args.query!r}\n")
    print(render_report(state.stages))
    for cluster in clusters:
        print(f"\ncluster {cluster.key}: {', '.join(cluster.paper_ids)}")

    if args.json_out:
        Path(args.json_out).write_text(state.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/pipeline/ -v`
Expected: PASS

- [ ] **Step 6: Run the CLI against the live service**

```bash
uv run python -m biolit.pipeline --query "metformin and lactic acidosis" --max-papers 20 --json-out /tmp/state.json
```

Expected: a stage ledger where the licence gate shows a non-zero permitted count with its "working as designed" note, and both stubs print `NOT IMPLEMENTED` with their notes. Paste the output into the commit message — it is the artifact this whole sub-project exists to produce.

If clusters come back empty, that is a legitimate outcome on a narrow query (spec §10 item 2). Record the observed count; **do not change `min_size`**, which is ADR-0013's validated setting.

- [ ] **Step 7: Full gate and commit**

```bash
uv run ruff format src/biolit/pipeline/ tests/pipeline/
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -q
grep -ciE '^name = "(nvidia|triton)' uv.lock   # must print 0
git add src/biolit/pipeline/ tests/pipeline/
git commit -F - <<'EOF'
feat(pipeline): end-to-end CLI with the stage ledger

`uv run python -m biolit.pipeline --query "..." --max-papers N` runs a
query through the real components: PubMed retrieval, NER,
canonicalization, the deterministic extractor (ADR-0015 rejected the LLM
one), and SameSentencePairing clustering (ADR-0013).

No LLM calls, no paid calls, no credential. Every component is local or a
free NCBI endpoint, so there is no pricing step and no authorization gate
because there is nothing to authorize.

The ledger is the deliverable, so the end-to-end test asserts it rather
than just the outputs -- and both not_implemented stages get their own
test against the JSON dump, being the requirement most likely to regress
silently.

Heavy imports stay local to main(), matching end_to_end.main. main() gets
no direct unit test; every function it calls is tested in its own module.

No run log: nothing here is a measured metric or a controlled experiment,
so the argument that justifies evals/*_runs.jsonl does not apply.
EOF
```

---

## Self-Review

**Spec coverage:**

| spec section | task |
|---|---|
| §0 measurement | context for Tasks 1–3; verified live in Task 3 Step 6 |
| §1 licence fix (URL→token, substring trap, batching, text-mining refusal) | Tasks 1–3 |
| §2 `PipelineState.stages` | Task 4 |
| §3 data flow | Tasks 5–7 |
| §4 drop ledger | Tasks 5, 6; rendering in Task 7 |
| §5 two unimplemented stages | Task 6; JSON-dump test in Task 7 |
| §6 CLI | Task 7 |
| §7 error handling (fail closed) | Task 3 (HTTP error), Task 2 (no prose), Task 5 (gate) |
| §8 testing (404 + stub cassettes, per-form fixtures, mutations) | Tasks 1–3, 5 |
| §9 rejected alternatives | no task — correctly, they are exclusions |
| §10 open items | Task 7 Step 6 records the cluster-count observation |

**Placeholders:** none — every code step carries runnable content.

**Type consistency:** `license_token_from_url` (Task 1) feeds `normalize_license` (existing) in Task 3. `licences_by_pmcid`/`normalize_pmcid` (Task 2) are consumed in Task 3. `StageReport`/`StageStatus` (Task 4) are used in Tasks 5–7. `RecordOutcome.records/licence/extract` (Task 5) are consumed in Task 7. `cluster_stage` returns `tuple[list[Cluster], StageReport]` consistently in Tasks 6 and 7.

**One ordering note for the executor:** Task 3 deletes two existing tests and modifies four others. If the suite is red between Steps 1 and 3 of that task, that is expected — it goes green at Step 4.
