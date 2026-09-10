"""Pin the modules whose behaviour determines a value a fixture displays.

⭐ THE RULE, so the set is derivable rather than remembered: pin every module whose behaviour
determines a value the fixture SHOWS. A fixture that claims "17 kept" after the ranker changed
is a false claim on a public page, and the requirement is that shipping one is structurally
hard rather than something anybody remembers to check.

⚠️ HASHED OVER THE PARSED AST WITH DOCSTRINGS STRIPPED, NOT OVER FILE BYTES. This repo's
modules carry very heavy comments and docstrings that are edited constantly; a byte hash would
fire on every prose edit, and a check that cries wolf gets suppressed. Normalising makes the
pin fire on behaviour and stay quiet on prose -- which is the correct trade, because a
comment-only edit genuinely does not invalidate a fixture.

⚠️ WHAT THIS DOES NOT COVER, listed rather than left unstated.

1. ~~`clients/pubmed.py::_parse_article`~~ ⛔ CLOSED 2026-09-10, after it bit. The module stays
   unpinned as a whole -- mostly transport -- but the functions that decide a paper's DOI, its
   licence lookup id and its abstract are now pinned by AST subtree (`PINNED_SUBTREES`), the
   upgrade path this entry had written down. It bit as DEF-0008: the parser read cited papers'
   identifiers as each paper's own, and five abstracts were quoted under borrowed licences
   while no pin could see the parser at all. `PubMedClient.efetch`'s own control flow remains
   unpinned transport.
2. `biolit_evals/fixture_export.py` itself. `project_run` determines displayed values -- the
   inf->None mapping, the concept-name lookup, rank numbering -- so by the rule above it
   qualifies. It is unpinned because pinning the module wholesale would also pin `main()`'s
   network plumbing; the recommended upgrade is to hash the `project_run` subtree alone, the
   same technique as (1). ⚠️ RECORDED AS AN OPEN GAP, not as a decision that it does not apply.
3. The MeSH ARTIFACTS (`mesh_tree.json.gz`, `mesh_actions.json.gz`, the alias dictionary) --
   data, not code. The traversal code is now pinned; the data it reads is not.
4. The NER checkpoint, and NCBI's query translation.

All of these are recorded only by `generated_at`; what catches them is regeneration.
"""

import ast
import hashlib
import importlib.util

#: Every module whose behaviour determines a value the fixture displays.
PINNED_MODULES: tuple[str, ...] = (
    "biolit.query.concepts",  # which clusters match
    "biolit.query.ranking",  # the order, and the score shown
    "biolit.pipeline.stages",  # every StageReport the site renders
    "biolit.cluster.pairing",  # which clusters exist at all
    "biolit.synth.template",  # renders the answer string, verbatim
    "biolit.domain.licensing",  # the tier table -> gate counts AND the tier shown
    "biolit.clients.pmc",  # parses the permissions block into the licence token
    # ⭐ ADDED 2026-09-09 — the pin catching up to its own rule, not new scope. The rule above
    # is self-verifying: "every module whose behaviour determines a value the fixture displays".
    # These five satisfy it and were simply missed when the list was first written. Treating the
    # list as fixed, rather than re-deriving it from the rule, is the exact failure the rule
    # exists to prevent.
    "biolit.extract.deterministic",  # chooses the sentences quoted verbatim in `answer`
    "biolit.extract.base",  # IS the licence gate: sets licence_gate's n_out, cuts the quotes
    "biolit.cluster.group",  # decides which clusters exist and their keys
    "biolit.canon.mesh_tree",  # `distance` produces the displayed `proximity`
    "biolit.canon.mesh_actions",  # `classes_of` produces the displayed `matched`
    # ADDED 2026-09-10 with the module itself: `FINDINGS` IS the callout copy a run page shows,
    # and `anchor_resolves` decides which callouts are allowed to ship.
    "biolit_evals.fixture_findings",
)

#: Modules too transport-heavy to pin whole, pinned by the SUBTREES that decide displayed values.
#: ⛔ ADDED 2026-09-10 (DEF-0008). `clients/pubmed.py` was left out entirely -- gap 1 below --
#: and its parser then read cited papers' identifiers as each paper's own, deciding DOIs and
#: licences on every fixture while no pin could see it. This is the upgrade path gap 1 wrote
#: down in advance: "hash that function's AST subtree alone".
PINNED_SUBTREES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "biolit.clients.pubmed",
        (
            "_OWN_ARTICLE_IDS",  # WHICH id list is the paper's own (DEF-0008)
            "_own_id",  # reads a paper's DOI and PMC id
            "_pmc_id_of",  # the id the licence lookup uses -> licence, tier, gate counts
            "PubMedClient._parse_article",  # title, journal, year, doi, pmid, licence
            "PubMedClient._parse_abstract",  # the text every quote in `answer` is cut from
        ),
    ),
)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                node.body = body[1:]
    return tree


def _normalise(source: str) -> str:
    """Source -> a string that changes with behaviour and not with prose."""
    return ast.dump(_strip_docstrings(ast.parse(source)))


def _find(body: list[ast.stmt], name: str) -> ast.stmt | None:
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                return node
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node
    return None


def _normalise_subtrees(source: str, names: tuple[str, ...]) -> str:
    """Normalise only the named nodes -- top-level names or `Class.method` -- in a fixed order.

    A name that cannot be found RAISES: a renamed function would otherwise drop out of the pin
    while the pin still claimed to cover it, the same shrinkage `source_pin` refuses.
    """
    tree = _strip_docstrings(ast.parse(source))
    assert isinstance(tree, ast.Module)
    parts = []
    for qualname in names:
        body: list[ast.stmt] = tree.body
        node: ast.stmt | None = None
        for part in qualname.split("."):
            node = _find(body, part)
            if node is None:
                raise RuntimeError(
                    f"fixture_pin: {qualname!r} not found. A pinned subtree was renamed or "
                    "removed without updating PINNED_SUBTREES -- refusing to pin less than claimed."
                )
            body = node.body if isinstance(node, ast.ClassDef) else []
        assert node is not None  # a qualname has at least one part, and a miss raised above
        parts.append(f"{qualname}={ast.dump(node)}")
    return "\n".join(parts)


def source_pin() -> str:
    """A hex digest over every pinned module's normalised source, in sorted order.

    Sorted rather than declaration order: a cosmetic reorder of the `PINNED_MODULES` tuple
    must not change the pin and force a needless fixture regeneration.
    """
    digest = hashlib.sha256()
    for name in sorted(PINNED_MODULES):
        digest.update(name.encode())
        digest.update(_normalise(_module_source(name)).encode())
    for name, qualnames in sorted(PINNED_SUBTREES):
        digest.update(f"{name}:{','.join(qualnames)}".encode())
        digest.update(_normalise_subtrees(_module_source(name), qualnames).encode())
    return digest.hexdigest()


def _module_source(name: str) -> str:
    # `find_spec` RAISES ModuleNotFoundError when the PARENT package is missing and returns
    # None when only the leaf is -- two shapes for the same mistake. Both mean the same thing
    # here, so both become the same refusal.
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"fixture_pin: cannot locate {name!r}. A pinned module was renamed or "
            "removed without updating PINNED_MODULES, which would silently reduce what "
            "the pin covers -- refusing rather than hashing a smaller set."
        )
    with open(spec.origin, encoding="utf-8") as handle:
        return handle.read()
