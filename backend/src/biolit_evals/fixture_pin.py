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

1. `clients/pubmed.py::_parse_article` assigns license, license_tier, doi and title, and is
   deliberately absent: that module is 263 lines dominated by transport that changes for
   reasons unrelated to what a fixture claims, so pinning it would fire constantly and a check
   that cries wolf gets suppressed. If it ever bites, hash that function's AST subtree alone.
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


def source_pin() -> str:
    """A hex digest over every pinned module's normalised source, in sorted order.

    Sorted rather than declaration order: a cosmetic reorder of the `PINNED_MODULES` tuple
    must not change the pin and force a needless fixture regeneration.
    """
    digest = hashlib.sha256()
    for name in sorted(PINNED_MODULES):
        # `find_spec` RAISES ModuleNotFoundError when the PARENT package is missing and
        # returns None when only the leaf is -- two shapes for the same mistake. Both mean the
        # same thing here, so both become the same refusal.
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
            digest.update(name.encode())
            digest.update(_normalise(handle.read()).encode())
    return digest.hexdigest()
