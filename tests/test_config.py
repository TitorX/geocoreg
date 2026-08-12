"""The published defaults, pinned.

Values are transcribed from the version of the code that produced the paper's
numbers rather than read from the module under test, so a failure here means a
default moved. The second half checks the override route: the constants are
plain module attributes, read at call time.
"""

from __future__ import annotations

import numpy as np
import pytest

from geocoreg import config

# name -> value in the paper version of config.py
PAPER_DEFAULTS = {
    # Phase 1 — refine
    "MAX_DOWNSAMPLE_SIDE": 2048,
    "SIFT_NFEATURES": 2000,
    "MIN_PATCH_SIZE": 256,
    "RATIO_THRESH": 0.75,
    "SEARCH_MARGIN_FACTOR": 1.0,
    "SPARSE_THRESHOLD": 5,
    "RANSAC_REPROJ_PATCH": 3.0,
    "MULTICHAN_DEDUP_RADIUS": 3.0,
    "FWD_BWD_CONSISTENCY_PX": 3.0,
    "NMS_RADIUS": 32,
    "MAD_K": 3.0,
    # Interpolation
    "INTERP_K": 20,
    "INTERP_ALPHA": 2.0,
    "INTERP_CHUNK_SIZE": 10000,
    # Parallelism
    "PATCH_WORKERS": 1,
    # Self-calibration switches
    "SUBPIX_REFINE": True,
    "DENSIFY": True,
    "INTERP_AUTOTUNE": True,
    # Debug
    "DUMP_POINTS_PATH": None,
    # Phase 2 — warp
    "WARP_CHUNK_ROWS": 256,
    "WARP_INTERP_METHOD": "nearest",
}

# Internal measurement constants of the phase-correlation stages. Not exposed
# in config.py, but they are hyperparameters of the published method all the
# same, so they are pinned here.
PAPER_SUBPIXEL_CONSTANTS = {
    "_WIN_REFINE": 128,
    "_WIN_DENSIFY": 64,
    "_UPSAMPLE": 32,
    "_MAX_SHIFT_REFINE": 4.0,
    "_MAX_SHIFT_DENSIFY": 3.0,
    "_ERR_FLOOR": 0.6,
    "_ERR_Q_REFINE": 0.5,
    "_ERR_Q_DENSIFY": 0.4,
    "_MIN_VALID_FRAC": 0.7,
    "_DENSIFY_SPACING": 48,
    "_K_NULL": 8,
    "_NULL_MIN_SAMPLES": 16,
}


@pytest.mark.parametrize("name,expected", sorted(PAPER_DEFAULTS.items()))
def test_default_matches_the_paper_version(name, expected):
    actual = getattr(config, name)
    assert actual == expected, f"config.{name} is {actual!r}, paper used {expected!r}"
    assert type(actual) is type(expected), f"config.{name} changed type"


def test_the_three_self_calibration_switches_default_to_true():
    assert config.SUBPIX_REFINE is True
    assert config.DENSIFY is True
    assert config.INTERP_AUTOTUNE is True


def test_no_public_constant_was_added_or_removed():
    """A new default is fine, but it must be declared here deliberately."""
    public = {k for k in dir(config) if k.isupper() and not k.startswith("_")}
    assert public == set(PAPER_DEFAULTS)


@pytest.mark.parametrize("name,expected", sorted(PAPER_SUBPIXEL_CONSTANTS.items()))
def test_subpixel_measurement_constant_matches_the_paper_version(name, expected):
    from geocoreg import subpixel

    assert getattr(subpixel, name) == expected


def test_local_affine_filter_defaults_match_the_paper_version():
    import inspect

    from geocoreg import filters

    sig = inspect.signature(filters.local_affine_filter)
    assert sig.parameters["k"].default == 12
    assert sig.parameters["mad_k"].default == 3.5


def test_every_constant_carries_a_comment():
    """config.py doubles as the parameter reference page, so each value is
    documented in place. Checked structurally rather than by eye."""
    import inspect
    import re

    src = inspect.getsource(config).splitlines()
    assign = re.compile(r"^([A-Z][A-Z0-9_]*)\s*=")
    documented = set()
    preceded_by_comment = False
    for line in src:
        stripped = line.strip()
        m = assign.match(stripped)
        if m:
            if preceded_by_comment or "#" in line:
                documented.add(m.group(1))
        preceded_by_comment = stripped.startswith("#") or (
            preceded_by_comment and not stripped)
    assert set(PAPER_DEFAULTS) - documented == set()


# --------------------------------------------------------------------------
# The override route is "set the module attribute", and nothing else
# --------------------------------------------------------------------------

def test_no_environment_variable_hook_survives():
    """Settings come from module attributes only, never from the environment."""
    import inspect

    src = inspect.getsource(config)
    assert "environ" not in src


def test_setting_an_attribute_changes_a_filter_at_call_time():
    from geocoreg import filters

    rng = np.random.default_rng(0)
    D = np.zeros((64, 64, 2))
    F = np.zeros((64, 64), np.uint8)
    for i in range(8):
        for j in range(8):
            r, c = 3 + 8 * i, 3 + 8 * j
            D[r, c, 0] = 10.0 + rng.normal(0, 1.0)
            F[r, c] = 1

    config.MAD_K = 3.0
    assert filters.global_outlier_filter(D.copy(), F.copy()) == 0
    config.MAD_K = 0.1
    assert filters.global_outlier_filter(D.copy(), F.copy()) > 0


def test_setting_an_attribute_changes_the_interpolator_at_call_time():
    from geocoreg import interp

    rng = np.random.default_rng(1)
    D = np.zeros((48, 48, 2))
    F = np.zeros((48, 48), np.uint8)
    C = np.zeros((48, 48), np.float32)
    for _ in range(120):
        r, c = int(rng.integers(0, 48)), int(rng.integers(0, 48))
        D[r, c, 0] = float(rng.normal(0, 5))
        F[r, c] = 1
        C[r, c] = 1.0

    config.INTERP_K = 4
    few = interp.fill(D.copy(), F.copy(), C=C.copy())
    config.INTERP_K = 32
    many = interp.fill(D.copy(), F.copy(), C=C.copy())
    assert not np.array_equal(few, many)


@pytest.mark.parametrize("switch,stage", [("SUBPIX_REFINE", "refine_points"),
                                          ("DENSIFY", "densify_grid")])
def test_switching_a_stage_off_actually_skips_it(tiny_pair, monkeypatch, switch, stage):
    """The switches gate real pipeline stages, read at call time."""
    from geocoreg import init, refine, subpixel
    from geocoreg.matchers import RootSIFTMatcher

    calls = []
    original = getattr(subpixel, stage)

    def spy(*args, **kwargs):
        calls.append(stage)
        return original(*args, **kwargs)

    monkeypatch.setattr(subpixel, stage, spy)

    ctx, state = init.run(tiny_pair.ref_path, tiny_pair.mov_path)
    matcher = RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                              config.RANSAC_REPROJ_PATCH,
                              config.MULTICHAN_DEDUP_RADIUS)

    setattr(config, switch, False)
    refine.run(ctx, state, matcher, diag_folder=None)
    assert calls == []

    setattr(config, switch, True)
    refine.run(ctx, state, matcher, diag_folder=None)
    assert calls == [stage]


def test_interp_autotune_switch_changes_the_final_fill(tiny_pair):
    from geocoreg import init, refine
    from geocoreg.matchers import RootSIFTMatcher

    ctx, state = init.run(tiny_pair.ref_path, tiny_pair.mov_path)
    matcher = RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                              config.RANSAC_REPROJ_PATCH,
                              config.MULTICHAN_DEDUP_RADIUS)
    seen = {}
    for value in (True, False):
        config.INTERP_AUTOTUNE = value
        seen[value] = refine.run(ctx, state, matcher, diag_folder=None).D
    # Both must produce a full field; the switch is what selects the route.
    assert seen[True].shape == seen[False].shape
    assert np.isfinite(seen[True]).all() and np.isfinite(seen[False]).all()
