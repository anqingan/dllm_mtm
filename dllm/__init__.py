"""Top-level package for dllm.

Subpackages are imported lazily so lightweight utilities such as
``dllm.duel`` do not force-load every model pipeline and optional backend.
"""

from importlib import import_module

__all__ = ["core", "data", "duel", "pipelines", "utils"]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
