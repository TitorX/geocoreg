"""Dense, per-pixel co-registration of georeferenced satellite imagery.

Reference implementation of SCDF (self-calibrating dense displacement fields),
Sun et al. 2026, arXiv:2608.22300.
"""

import logging

try:
    from osgeo import gdal as _gdal  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "geocoreg requires GDAL's Python bindings (osgeo), which are not "
        "installed. GDAL is not reliably installable with pip; install it "
        "from conda-forge (`conda install -c conda-forge gdal`) or with pixi, "
        "then reinstall geocoreg. See README.md."
    ) from exc

__version__ = "1.0.0"

__all__ = ["__version__", "run_match", "run_apply"]

# A library must not configure the root logger; the CLI attaches its own
# handlers via pipeline.setup_logging().
logging.getLogger("geocoreg").addHandler(logging.NullHandler())


def __getattr__(name):
    """Resolve the public entry points lazily (PEP 562).

    Importing them eagerly would pull the whole pipeline (and its heavy
    dependencies) into every ``import geocoreg.config``, and ``pipeline``
    itself imports from this package.
    """
    if name in ("run_match", "run_apply"):
        from geocoreg import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + ["run_match", "run_apply"])
