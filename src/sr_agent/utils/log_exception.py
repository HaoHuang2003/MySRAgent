# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
import traceback
from .tag2ansi import tag2ansi


def log_exception(e: Exception) -> str:
    """Format exception for logging."""
    return tag2ansi(
        f"[red bold]{type(e).__name__}[reset]: "
        f"[red]{str(e)}[reset]\n"
        f"[gray]{traceback.format_exc()}[reset]"
    )
