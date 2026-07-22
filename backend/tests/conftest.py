from pathlib import Path

import pytest

# Repo root: this file is <repo>/backend/tests/conftest.py
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Point the tdd-guard pytest reporter at the repo-root ``.claude/`` directory.

    The reporter resolves its storage from the ``tdd_guard_project_root`` ini option,
    which it requires to be an *absolute* path. We can't hardcode one in pyproject.toml
    without breaking every other machine and CI, so we compute the repo root here and
    set the plugin's storage directory directly.

    Without this, running pytest from ``backend/`` writes the guard's state to
    ``backend/.claude/`` while the guard hook reads ``<repo>/.claude/``, so the guard
    sees perpetually stale test results. No-op when the plugin isn't installed.
    """
    try:
        from tdd_guard_pytest.pytest_reporter import tdd_guard_stash_key
    except ImportError:
        return

    plugin = config.stash.get(tdd_guard_stash_key, None)
    if plugin is not None:
        plugin.storage_dir = _REPO_ROOT / ".claude" / "tdd-guard" / "data"
