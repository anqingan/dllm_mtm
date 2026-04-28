"""Pipeline subpackages.

Pipeline modules are imported lazily so optional or version-sensitive model
dependencies in one pipeline do not affect unrelated pipelines.
"""

from importlib import import_module

__all__ = [
    "a2d",
    "bert",
    "dream",
    "editflow",
    "fastdllm",
    "llada",
    "llada2",
    "llada21",
    "rl",
]


def __getattr__(name):
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
