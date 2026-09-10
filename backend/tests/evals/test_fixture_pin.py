import pytest

from biolit_evals.fixture_pin import _normalise, source_pin


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
