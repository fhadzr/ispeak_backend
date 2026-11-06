"""
app/services/ispeak/language_tool.py

Lazy, process-wide LanguageTool singleton.

LanguageTool spawns a Java subprocess; constructing it per request is expensive
and, when Java is unavailable, silently swallowed the failure (see audit H-4).
This module initialises it once on first use, logs availability explicitly, and
never hides the reason it is unavailable.
"""

import logging
import threading

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_tool = None
_init_attempted = False


def _build_tool():
    try:
        import language_tool_python

        return language_tool_python.LanguageTool("en-US")
    except Exception as e:
        logger.error(
            "LanguageTool tidak tersedia (Java mungkin belum terinstal): %s. "
            "Skor grammar_errors akan menjadi 0.",
            e,
        )
        return None


def get_language_tool():
    """Return a shared LanguageTool instance, or None if unavailable."""
    global _tool, _init_attempted

    if _init_attempted:
        return _tool

    with _LOCK:
        if not _init_attempted:
            _tool = _build_tool()
            _init_attempted = True
            if _tool is not None:
                logger.info("LanguageTool siap digunakan.")

    return _tool


def check_grammar(text: str) -> int:
    """Return the number of grammar errors, or 0 when LanguageTool is unavailable.

    Unavailability is logged once (in ``get_language_tool``) and does not raise.
    """
    tool = get_language_tool()
    if tool is None:
        return 0
    try:
        return len(tool.check(text))
    except Exception as e:
        logger.error("LanguageTool check gagal: %s", e)
        return 0


def shutdown_language_tool():
    """Close the shared tool, if any. Called from the app shutdown hook."""
    global _tool, _init_attempted
    with _LOCK:
        if _tool is not None:
            try:
                _tool.close()
            except Exception as e:
                logger.warning("Gagal menutup LanguageTool: %s", e)
        _tool = None
        _init_attempted = False
