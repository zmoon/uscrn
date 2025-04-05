import datetime
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


def current_commit() -> str | None:
    """Get the current commit hash (short version) or return ``None``."""
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
    """Get the tags and their associated commit hashes (short versions) or return ``None``."""
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
    """Check if we are on ReadTheDocs by looking for READTHEDOCS=True in the environment."""
    import os

    return os.environ.get("READTHEDOCS", "False") == "True"


def commit_date(commit: str) -> datetime.datetime | None:
    """Get the date/time of a commit or return ``None``."""
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


def maybe_fancy_version(package: str, /) -> str:
    """Attempt to embellish the version string with commit hash and date.
    Otherwise, return the detected version string as is.
    Special treatment for ReadTheDocs builds.
    """
    import os

    set_version = getattr(package, "__version__", None)
    if set_version is None:
        logger.info(f"Package {package} does not have a __version__ attribute")
        try:
            from importlib.metadata import PackageNotFoundError, version
        except ImportError:
            logger.warning("Failed to import importlib.metadata")
            set_version = "?"
        else:
            try:
                set_version = version(package)
            except PackageNotFoundError:
                logger.warning(f"Package {package} not found")
                set_version = "?"
    logger.debug(f"Package {package} detected version: {set_version}")

    commit: str | None
    if on_rtd():
        rtd_git_id = os.environ["READTHEDOCS_GIT_IDENTIFIER"]
        if rtd_git_id == set_version:
            return set_version
        else:
            rtd_git_hash = os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
            commit = rtd_git_hash[:7]

            ver = f"{set_version}+{commit}"
            date = commit_date(commit)
            if date is not None:
                ver += f" ({date:%Y-%m-%d})"

            return ver

    commit = current_commit()
    if commit is None:
        return set_version

    tags = get_tags()
    if tags is None:
        return set_version  # or tag?

    for tag, tag_commit in tags:
        if tag_commit == commit:
            logger.debug(f"Commit {commit} is tagged ({tag})")
            return set_version

    ver = f"{set_version}+{commit}"
    date = commit_date(commit)
    if date is not None:
        ver += f" ({date:%Y-%m-%d})"

    return ver
