from __future__ import annotations

import datetime
import logging
from pathlib import Path

HERE = Path(__file__).parent

logger = logging.getLogger("uscrn")


def retry(func):
    """Decorator to retry a function on web connection error.
    Up to 60 s, with Fibonacci backoff (1, 1, 2, 3, ...).
    """
    import urllib
    from functools import wraps

    import requests

    max_time = 60_000_000_000  # 60 s (in ns)

    @wraps(func)
    def wrapper(*args, **kwargs):
        from time import perf_counter_ns, sleep

        t0 = perf_counter_ns()
        a, b = 1, 1
        while True:
            try:
                return func(*args, **kwargs)
            except (
                urllib.error.URLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                if perf_counter_ns() - t0 > max_time:  # pragma: no cover
                    raise
                logger.info(
                    f"Retrying {func.__name__} in {a} s after connection error",
                    stacklevel=2,
                )
                sleep(a)
                a, b = b, a + b  # Fibonacci backoff

    return wrapper


def current_commit() -> str | None:
    import subprocess

    maybe_repo = HERE.parent

    cmd = ["git", "-C", maybe_repo.as_posix(), "rev-parse", "--verify", "--short", "HEAD"]
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True)
    except Exception:
        return None
    else:
        return cp.stdout.strip()


def get_tags() -> list[tuple[str, str]] | None:
    import subprocess

    maybe_repo = HERE.parent

    cmd = ["git", "-C", maybe_repo.as_posix(), "tag"]
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True)
    except Exception:
        return None
    else:
        tags = cp.stdout.strip().splitlines()

    # Get commit associated with each tag
    commits = []
    for tag in tags:
        cmd = ["git", "-C", maybe_repo.as_posix(), "rev-list", "-n", "1", "--abbrev-commit", tag]
        try:
            cp = subprocess.run(cmd, text=True, capture_output=True)
        except Exception:
            commit = None
        else:
            commit = cp.stdout.strip()
        commits.append(commit)

    return list(zip(tags, commits))


def commit_date(commit: str) -> datetime.datetime | None:
    import subprocess

    maybe_repo = HERE.parent

    cmd = ["git", "-C", maybe_repo.as_posix(), "show", "--no-patch", r"--format=%cI", commit]
    try:
        cp = subprocess.run(cmd, text=True, capture_output=True)
    except Exception:
        return None
    else:
        iso = cp.stdout.strip()

    return datetime.datetime.fromisoformat(iso)


def maybe_fancy_version() -> str:
    from . import __version__

    commit = current_commit()
    if commit is None:
        return __version__

    tags = get_tags()
    if tags is None:
        return __version__

    for _, tag_commit in tags:
        if tag_commit == commit:
            return __version__

    ver = f"{__version__}+{commit}"
    date = commit_date(commit)
    if date is not None:
        ver += f" ({date:%Y-%m-%d})"

    return ver
