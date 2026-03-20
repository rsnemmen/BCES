"""BCES linear regression for data with measurement errors."""

__version__ = "2.0"

# NOTE: `from bces import bces` returns the bces.bces submodule (not the function)
# due to a naming collision — the package, submodule, and core function share the
# name "bces". Use `import bces.bces as BCES; BCES.bces(...)` or
# `from bces.bces import bces` to access the regression function directly.

__all__ = ["bces", "bcesboot", "bcesp", "bootstrap", "wls", "wlsboot", "wlsp"]


def __getattr__(name):
    """Lazy attribute access for non-submodule names."""
    if name in ("bcesboot", "bcesp", "bootstrap", "wls", "wlsboot", "wlsp"):
        import bces.bces as _mod
        return getattr(_mod, name)
    raise AttributeError(f"module 'bces' has no attribute {name!r}")
