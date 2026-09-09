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
