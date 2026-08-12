"""Shared fixtures and markers for the geocoreg test suite.

Unit tests over pure functions run by default; the small end-to-end pipeline
runs are marked ``slow``. No imagery ships with the suite — every input is
generated procedurally by ``tests/_synth.py``.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ensure_proj_data():
    """Point PROJ at its database when the environment has not.

    GDAL installed without its activation scripts finds no ``proj.db`` and
    every CRS transform warns, which drowns the test report.
    """
    if os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB"):
        return
    candidates = [
        os.path.join(sys.prefix, "share", "proj"),
        os.path.join(sys.base_prefix, "share", "proj"),
    ]
    try:
        import osgeo

        candidates.append(
            os.path.join(os.path.dirname(os.path.dirname(osgeo.__file__)),
                         "..", "..", "share", "proj"))
    except ImportError:  # pragma: no cover - geocoreg cannot import without it
        pass
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "proj.db")):
            os.environ["PROJ_DATA"] = os.path.normpath(cand)
            return


_ensure_proj_data()


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end pipeline run")


@pytest.fixture(autouse=True)
def config_snapshot():
    """Restore ``geocoreg.config`` after every test — it is process-global."""
    from geocoreg import config

    saved = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    yield config
    for k, v in saved.items():
        setattr(config, k, v)


@pytest.fixture(autouse=True)
def quiet_logging():
    """Keep the pipeline's INFO chatter out of the test report."""
    logger = logging.getLogger("geocoreg")
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.CRITICAL)
    yield
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)


@pytest.fixture(autouse=True)
def _no_matplotlib_windows():
    """Force a headless matplotlib backend when the optional extra is present."""
    try:
        import matplotlib
    except ImportError:
        return
    matplotlib.use("Agg", force=True)


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("geocoreg")


@pytest.fixture(scope="session")
def tiny_pair(session_tmp):
    """A 192 px same-CRS pair: too small for any refine level, so the whole
    pipeline runs in a fraction of a second. Used for switch/plumbing tests."""
    import _synth

    return _synth.make_pair(str(session_tmp), size=192,
                            d_fn=_synth.d_constant(30.0, -20.0), name="tiny")


@pytest.fixture(scope="session")
def small_pair(session_tmp):
    """A 256 px same-CRS, same-GSD pair with a pure translation."""
    import _synth

    return _synth.make_pair(str(session_tmp), size=256,
                            d_fn=_synth.d_constant(35.0, -25.0), name="small")
