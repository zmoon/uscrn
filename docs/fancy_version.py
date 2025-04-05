from __future__ import annotations

import datetime
import importlib
import inspect
import logging
import os
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from packaging.version import Version

logger = logging.getLogger(__name__)

__all__ = [
    "GitInfo",
    "TagInfo",
    "VersionInfo",
    "logger",
]


class TagInfo(NamedTuple):
    name: str
    """Tag name."""

    commit: str | None
    """Associated commit hash (short version) or ``None`` if commit detection failed."""


class GitInfo:
    """Class that handles Git repository information.

    Parameters
    ----------
    directory
        Path to a directory within a (potential) Git repository.
    """

    def __init__(self, directory: str | Path, /) -> None:
        self.repo = self._find_git_repo(Path(directory))
        """The detected Git repository path.
        Errors aren't raised if this isn't actually a Git repository.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.repo})"

    def __str__(self) -> str:
        return self.repo.as_posix()

    @staticmethod
    @lru_cache(1)
    def on_rtd() -> bool:
        """Check if we are on ReadTheDocs."""
        return os.environ.get("READTHEDOCS", "False") == "True"

    def _find_git_repo(self, start_path: Path, /) -> Path:
        """Find the Git repository by walking up the directory tree.

        Parameters
        ----------
        start_path
            Path to start searching from.

        Returns
        -------
        :
            Path to the Git repository root or the start path if no repo found.
        """
        if self.on_rtd():
            return Path(os.environ["READTHEDOCS_REPOSITORY_PATH"])

        for path in [start_path, *start_path.parents]:
            if path.name == "site-packages":
                break
            if (path / ".git").is_dir():
                return path

        logger.debug("Could not find .git directory, using provided directory")
        return start_path

    def current_commit(self) -> str | None:
        """Get the current commit hash (short version).

        Returns
        -------
        :
            The short commit hash or ``None`` if it couldn't be retrieved.
        """
        if self.on_rtd():
            rtd_git_hash = os.environ["READTHEDOCS_GIT_COMMIT_HASH"]
            return rtd_git_hash[:7]

        cmd = ["git", "-C", self.repo.as_posix(), "rev-parse", "--verify", "--short", "HEAD"]
        try:
            cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except Exception:
            logger.exception("Could not get commit hash")
            return None
        else:
            return cp.stdout.strip()

    def commit_date(self, commit: str) -> datetime.datetime | None:
        """Get the date/time of a commit.

        Parameters
        ----------
        commit
            The commit hash of interest.
        """
        cmd = ["git", "-C", self.repo.as_posix(), "show", "--no-patch", r"--format=%cI", commit]
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

    def tags(self) -> list[TagInfo] | None:
        """Get the tags and their associated commit hashes.

        Returns
        -------
        :
            List of TagInfo(name, commit) objects or ``None`` if tags couldn't be retrieved.
        """
        cmd = ["git", "-C", self.repo.as_posix(), "tag"]
        try:
            cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except Exception:
            logger.exception("Could not get tags")
            return None
        else:
            tags = cp.stdout.strip().splitlines()

        # Get commit associated with each tag
        result = []
        for tag in tags:
            cmd = ["git", "-C", self.repo.as_posix(), "rev-list", "-n", "1", "--abbrev-commit", tag]
            try:
                cp = subprocess.run(cmd, text=True, capture_output=True)
            except Exception:
                logger.exception(f"Could not get commit for tag {tag}")
                commit = None
            else:
                commit = cp.stdout.strip()
            result.append(TagInfo(tag, commit))

        return result

    def dirty(self) -> bool:
        """Are there staged or unstaged changes?"""
        if self.on_rtd():
            return False

        cmd = ["git", "-C", self.repo.as_posix(), "status", "--porcelain"]
        try:
            cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except Exception:
            logger.exception("Could not get dirty status")
            return False
        else:
            return cp.stdout.strip() != ""


class VersionInfo:
    """Class to handle version information for a package, including Git metadata.

    Parameters
    ----------
    package
        Either a string package name (to be passed to :func:`import_module`)
        or an actual module object.
    """

    def __init__(self, package: str | ModuleType) -> None:
        if isinstance(package, str):
            self.module = importlib.import_module(package)
        elif isinstance(package, ModuleType):
            self.module = package
        else:
            raise TypeError(f"Package must be a string or a module object. Got {type(package)}.")

        spec = self.module.__spec__
        if spec is None:
            raise RuntimeError(f"Module spec is None for module {self.package_name!r}.")
        if not (spec.name == spec.parent == self.package_name):
            raise ValueError(f"Module {self.package_name!r} is not a top-level module.")

        self.git_info = GitInfo(Path(inspect.getfile(self.module)).parent)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.module})"

    def __str__(self):
        return self.package_name

    @property
    def package_name(self) -> str:
        """The package name, which works for importing, e.g. 'numpy'."""
        return self.module.__name__

    def version(self) -> str:
        """Get the package version.

        Returns
        -------
        str
            The package version string, first trying ``.__version__`` and then
            using :func:`importlib.metadata.version()`.
        """
        set_version = getattr(self.module, "__version__", None)

        if set_version is None:
            logger.info(f"Package {self.package_name!r} does not have a __version__ attribute")
            try:
                set_version = version(self.package_name)
            except PackageNotFoundError:
                logger.warning(f"Package {self.package_name!r} not found")
                set_version = "?"

        logger.debug(f"Package {self.package_name!r} detected version: {set_version}")
        return set_version

    def version_info(self) -> Version:
        """Parse the package version, returning a :class:`packaging.version.Version` object."""
        from packaging.version import Version

        return Version(self.version())

    def on_tag(self) -> bool | None:
        """Are we a tagged commit? Or ``None`` if can't determine."""
        if self.git_info.on_rtd():
            rtd_git_id = os.environ["READTHEDOCS_GIT_IDENTIFIER"]
            logger.debug(f"READTHEDOCS_GIT_IDENTIFIER: {rtd_git_id}")
            return rtd_git_id == self.version()

        tags = self.git_info.tags()
        if tags is None:
            return None

        commit = self.git_info.current_commit()
        if commit is None:
            return None

        for tag_info in tags:
            if tag_info.commit == commit:
                logger.debug(f"Commit {commit} is tagged ({tag_info.name})")
                return True

        return False

    def fancy_version(
        self,
        *,
        commit_template: str = "{commit}",
    ) -> str:
        """Generate a fancy version string with Git information if available.

        Parameters
        ----------
        commit_template
            Template for the commit hash, to use with ``.format()``.
            In case you want to style it, e.g. linking to GitHub.

        Returns
        -------
        :
            The enhanced version string with Git information when available.
        """
        set_version = self.version()

        on_tag = self.on_tag()
        if on_tag or on_tag is None:
            return set_version

        commit = self.git_info.current_commit()
        assert isinstance(commit, str)
        ver = f"{set_version}+{commit_template.format(commit=commit)}"
        if self.git_info.dirty():
            ver += "*"
        date = self.git_info.commit_date(commit)
        if date is not None:
            ver += f" ({date:%Y-%m-%d})"

        return ver
