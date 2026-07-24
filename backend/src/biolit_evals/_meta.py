import subprocess


def git_sha() -> str:
    """Short HEAD SHA for run-log provenance. Best-effort: never fail an eval over it."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"
