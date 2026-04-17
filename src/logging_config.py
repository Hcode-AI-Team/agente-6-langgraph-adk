"""Centralised logging configuration for the entire application.

All modules obtain their logger via `get_logger(__name__)`, which calls
`setup_logging()` exactly once (guaranteed by lru_cache).  This keeps log
format and level consistent regardless of which module is imported first.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings

_console = Console(stderr=True)


@lru_cache(maxsize=1)
def setup_logging() -> None:
    """Configure the root logger with a single stderr handler.

    lru_cache makes this idempotent: repeated calls from different modules
    during import time do not add duplicate handlers.

    The log level comes from Settings, but setup_logging() is called at module
    import time (when `get_logger(__name__)` is used at module level).  In test
    environments the required env vars may not yet be set, so we gracefully fall
    back to INFO rather than crashing during collection.
    """
    log_level = "INFO"
    try:
        log_level = get_settings().log_level
    except Exception:  # noqa: BLE001
        # Missing or invalid env vars during import-time logging setup must not
        # prevent modules from loading.  The real validation happens in main.py.
        pass

    root = logging.getLogger()
    root.setLevel(log_level)

    # Skip adding a second handler if the caller (e.g. a test runner) has
    # already configured the root logger.
    if root.handlers:
        return

    handler = RichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
        omit_repeated_times=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.setFormatter(logging.Formatter(fmt="%(name)s | %(message)s"))
    root.addHandler(handler)

    # Third-party libraries that are verbose at DEBUG level — suppress them
    # unless the user explicitly sets LOG_LEVEL=DEBUG for those namespaces.
    for noisy_lib in ("google.auth", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    setup_logging()
    return logging.getLogger(name)
