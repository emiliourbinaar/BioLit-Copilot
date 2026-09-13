import ast

import pytest

from biolit_evals.fixture_pin import (
    PINNED_SUBTREES,
    _canonical,
    _normalise,
    _normalise_subtrees,
    source_pin,
)


def test_the_pin_ignores_comments_and_docstrings_but_not_statements():
    """⭐ WHY THIS NORMALISES INSTEAD OF HASHING BYTES, and it is not a micro-optimisation.

    This repo's modules carry very heavy comments and docstrings, edited constantly and
    deliberately -- several of them are the only record of a fixed defect. A byte hash would
    fire on every prose edit, and a staleness check that cries wolf is a check that gets
    suppressed, which is worse than not having one.

    So the pin must be blind to comments and docstrings and sensitive to behaviour. A
    comment-only change genuinely does NOT invalidate a fixture; a changed statement does.
    """
    commented = '"""Doc one."""\n# a comment\nX = 1\n'
    reworded = '"""Doc two, entirely rewritten."""\n# a different comment\nX = 1\n'
    behavioural = '"""Doc one."""\n# a comment\nX = 2\n'

    assert _normalise(commented) == _normalise(reworded)
    assert _normalise(commented) != _normalise(behavioural)


def test_source_pin_is_stable_across_calls_and_refuses_a_missing_module(monkeypatch):
    """Stability first: a pin recomputed in the same tree must be identical, or every fixture
    would read stale on every run and the check would be worthless.

    And a renamed module must RAISE rather than hash a smaller set. Silently pinning six
    modules instead of seven is the failure mode that matters here -- the check would still
    pass, while covering less than its docstring claims.
    """
    assert source_pin() == source_pin()

    # Both shapes of "gone", because `find_spec` reports them differently: a missing LEAF
    # returns None, while a missing PARENT package raises ModuleNotFoundError. Verified against
    # the real interpreter -- find_spec("biolit.no.such") raises, find_spec(
    # "biolit.query.no_such_module") returns None -- so testing only one would leave the other
    # path unexercised and the pin silently able to crash instead of refusing.
    for gone in ("biolit.query.no_such_module", "biolit.no.such"):
        monkeypatch.setattr(
            "biolit_evals.fixture_pin.PINNED_MODULES", ("biolit.query.ranking", gone)
        )
        with pytest.raises(RuntimeError, match="A pinned module was renamed"):
            source_pin()


def test_docstrings_are_stripped_inside_classes_and_functions_too_not_only_at_module_level():
    """⚠️ THE COVERAGE GAP THIS CLOSES. The first test used a MODULE-level docstring only, so
    dropping `ClassDef`, `FunctionDef` or `AsyncFunctionDef` from `_strip_docstrings`'s
    isinstance tuple would still have passed CI — while silently changing every pin and forcing
    a live-NCBI regeneration on the next prose edit inside a class or function.

    Verifying the code is correct today is not the same as guarding it against tomorrow, which
    is exactly the distinction this repo keeps having to relearn.
    """
    source = (
        'class C:\n    """Class doc."""\n\n'
        '    def m(self):\n        """Method doc."""\n        return 1\n\n'
        'async def f():\n    """Async doc."""\n    return 2\n'
    )
    reworded = (
        'class C:\n    """Entirely different class prose."""\n\n'
        '    def m(self):\n        """Entirely different method prose."""\n        return 1\n\n'
        'async def f():\n    """Entirely different async prose."""\n    return 2\n'
    )
    behavioural = source.replace("return 1", "return 99")

    assert _normalise(source) == _normalise(reworded)
    assert _normalise(source) != _normalise(behavioural)


def test_a_pinned_subtree_fires_on_its_own_behaviour_and_nowhere_else():
    """DEF-0008's lesson, applied to the pin. `clients/pubmed.py` was left unpinned because
    most of it is transport that changes for reasons unrelated to what a fixture shows -- and
    the parser inside it then silently read a cited paper's identifiers as each paper's own.
    The written upgrade path was to hash the parsing functions' AST subtrees alone.

    So: a behaviour change inside a named function or constant must change the digest; the
    same change in an UNNAMED sibling must not; prose must not; and a name that no longer
    exists must raise rather than quietly pin less than it claims to.
    """
    source = (
        'IDS = "PubmedData/ArticleIdList/ArticleId"\n'
        "def transport():\n    return 'retry'\n"
        "class Client:\n"
        '    def _parse(self, x):\n        """Doc."""\n        return x.find(IDS)\n'
    )
    names = ("IDS", "Client._parse")

    base = _normalise_subtrees(source, names)
    assert base == _normalise_subtrees(source.replace('"""Doc."""', '"""Other."""'), names)
    assert base == _normalise_subtrees(source.replace("'retry'", "'backoff'"), names)
    assert base != _normalise_subtrees(source.replace("x.find", "x.findall"), names)
    assert base != _normalise_subtrees(source.replace("PubmedData/", ".//"), names)
    with pytest.raises(RuntimeError, match="Client._gone"):
        _normalise_subtrees(source, ("Client._gone",))


def test_the_canonical_form_ignores_default_valued_fields_a_newer_python_leaves_out():
    """⛔ THE BUG THIS REPLACES. The pin hashed `ast.dump`, whose output is a CPython
    implementation detail: Python 3.13 stopped printing empty-list and None fields, so the same
    source hashed differently on 3.12 and 3.14 and every fixture read STALE on a clean checkout.

    A newer interpreter is simulated here, in one interpreter, by deleting every default-valued
    field -- the exact difference observed. The canonical form must not change, while a real
    behavioural change still must.
    """
    source = "def f(x, *, y=1):\n    return [x, y]\n"
    full = ast.parse(source)
    pruned = ast.parse(source)
    for node in ast.walk(pruned):
        for field in list(node._fields):
            value = getattr(node, field, None)
            if value is None or (isinstance(value, list) and not value):
                if hasattr(node, field):
                    delattr(node, field)

    assert _canonical(full) == _canonical(pruned)
    assert _canonical(full) != _canonical(ast.parse(source.replace("[x, y]", "[y, x]")))
    assert _canonical(ast.parse("x = 1")) != _canonical(ast.parse("x = True"))
    assert _canonical(ast.parse("x = 1")) != _canonical(ast.parse("x = '1'"))


def test_source_pin_never_consults_ast_dump(monkeypatch):
    """The canonical form is only interpreter-independent if nothing reaches back into
    `ast.dump`. A later "simplification" back to it would pass every other test on whichever
    single interpreter CI runs, so this makes any call to it fail loudly instead.
    """

    def refuse(*_args, **_kwargs):
        raise AssertionError("source_pin must not depend on ast.dump's output format")

    monkeypatch.setattr(ast, "dump", refuse)
    assert len(source_pin()) == 64


def test_source_pin_covers_the_pubmed_parsing_subtrees_and_refuses_one_that_vanished(monkeypatch):
    """The subtrees must be IN the pin, not merely hashable. Pointing `PINNED_SUBTREES` at a
    name that does not exist must make `source_pin` refuse, which it can only do if it reads
    the list; and the real list must name the functions that decide a paper's DOI, licence
    lookup id and quoted abstract text.
    """
    pinned = dict(PINNED_SUBTREES)["biolit.clients.pubmed"]
    for name in (
        "_own_id",
        "_pmc_id_of",
        "PubMedClient._parse_article",
        "PubMedClient._parse_abstract",
    ):
        assert name in pinned
    assert source_pin() == source_pin()

    monkeypatch.setattr(
        "biolit_evals.fixture_pin.PINNED_SUBTREES",
        (("biolit.clients.pubmed", ("PubMedClient._renamed_away",)),),
    )
    with pytest.raises(RuntimeError, match="_renamed_away"):
        source_pin()
