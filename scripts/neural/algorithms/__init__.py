# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import importlib
from pathlib import Path


def list_algorithms() -> list[str]:
    return sorted(
        path.stem for path in Path(__file__).parent.glob("*.py")
        if path.stem != "__init__" and not path.stem.startswith("_")
    )


def get_algorithm(name: str):
    return importlib.import_module(f".{name}", package=__name__)


def update_parser(name: str, parser):
    module = get_algorithm(name)
    if hasattr(module, "update_parser"):
        parser = module.update_parser(parser)
    return parser
