from __future__ import annotations

import importlib


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is missing."""


def require_dependency(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"Missing optional dependency '{module_name}'. Install project dependencies first."
        ) from exc

