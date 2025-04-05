import datetime
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


def current_commit() -> str | None:
    import subprocess

    maybe_repo = HERE.parent

    cmd = ["git", "-C", maybe_repo.as_posix(), "rev-parse", "--verify", "--short", "HEAD"]
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        logger.exception("Could not get commit hash")
        return None
    else:
        return cp.stdout.strip()


def get_tags() -> list[tuple[str, str | None]] | None:
    import subprocess

    maybe_repo = HERE.parent

    cmd = ["git", "-C", maybe_repo.as_posix(), "tag"]
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        logger.exception("Could not get tags")
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
            logger.exception(f"Could not get commit for tag {tag}")
            commit = None
        else:
            commit = cp.stdout.strip()
        commits.append(commit)

    return list(zip(tags, commits))


@lru_cache(1)
def on_rtd() -> bool:
    import os

    return os.environ.get("READTHEDOCS", "False") == "True"


def commit_date(commit: str) -> datetime.datetime | None:
    import subprocess

    if on_rtd():
        import os

        maybe_repo = Path(os.environ["READTHEDOCS_REPOSITORY_PATH"])
    else:
        maybe_repo = HERE.parent

    cmd = ["git", "-C", maybe_repo.as_posix(), "show", "--no-patch", r"--format=%cI", commit]
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except Exception:
        logger.exception(f"Could not get date for commit {commit}")
        return None
    else:
        iso = cp.stdout.strip()

    if iso == "":
        logger.debug(f"Git returned empty string for commit {commit} date")
        return None

    return datetime.datetime.fromisoformat(iso)


def maybe_fancy_version() -> str:
    import os

    from . import __version__

    commit: str | None
    if on_rtd():
        rtd_git_id = os.environ["READTHEDOCS_GIT_IDENTIFIER"]
        if rtd_git_id == __version__:
            return __version__
        else:
            rtd_git_hash = os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
            commit = rtd_git_hash[:7]

            ver = f"{__version__}+{commit}"
            date = commit_date(commit)
            if date is not None:
                ver += f" ({date:%Y-%m-%d})"

            return ver

    commit = current_commit()
    if commit is None:
        return __version__

    tags = get_tags()
    if tags is None:
        return __version__  # or tag?

    for tag, tag_commit in tags:
        if tag_commit == commit:
            logger.debug(f"Commit {commit} is tagged ({tag})")
            return __version__

    ver = f"{__version__}+{commit}"
    date = commit_date(commit)
    if date is not None:
        ver += f" ({date:%Y-%m-%d})"

    return ver
