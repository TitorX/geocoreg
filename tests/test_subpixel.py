"""The phase-correlation measurement stages and their self-calibrating gates.

Two claims: the correlation path really measures a sub-pixel shift (the stage
exists to replace the feature matcher's ~0.3-0.5 px localization), and the
acceptance gate is *relative* — a window is accepted for beating its own null
distribution, so a mismatched partner is rejected however good its absolute
correlation score looks.
"""

from __future__ import annotations

import _synth
import cv2
import numpy as np
import pytest

from geocoreg import config, init, subpixel

WIN = 128
GSD = 10.0


def _band_limited(n=WIN, seed=0, sigma=2.0):
    """A texture with no energy near Nyquist, so a sub-pixel shift is defined."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(0, 1, (n, n)), (0, 0), sigmaX=sigma)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img.astype(np.float32)


def _fourier_shift(img, dy, dx):
    """Move the content by ``(+dy, +dx)`` pixels, exactly (band-limited image)."""
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    spec = np.fft.fft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx))
    return np.real(np.fft.ifft2(spec)).astype(np.float32)


# --------------------------------------------------------------------------
# the measurement itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dy,dx", [(0.0, 0.0), (0.25, 0.0), (0.0, -0.25),
                                   (0.375, -0.5), (-0.625, 0.25), (0.5, 0.5)])
def test_phase_correlation_recovers_a_subpixel_shift(dy, dx):
    """Recovery to better than 0.05 px, and the sign convention pinned.

    ``_correlate`` returns the shift that registers MOV onto REF, i.e. the
    negative of how far the MOV content moved. ``_finalize`` adds that shift to
    the REF window center, so flipping the sign would double every correction
    instead of removing it.
    """
    ref = _band_limited(seed=1)
    mov = _fourier_shift(ref, dy, dx)
    s_row, s_col, err, response = subpixel._correlate(ref, mov, WIN)

    assert abs(s_row - (-dy)) < 0.05, f"row: got {s_row}, want {-dy}"
    assert abs(s_col - (-dx)) < 0.05, f"col: got {s_col}, want {-dx}"
    assert 0.0 <= err < 0.2
    assert response > 0.8


def test_recovered_shifts_are_quantised_by_the_upsample_factor():
    """Precision is ~1/_UPSAMPLE of a pixel, by construction."""
    step = 1.0 / subpixel._UPSAMPLE
    ref = _band_limited(seed=2)
    for dy, dx in [(0.3, -0.2), (0.11, 0.44)]:
        s_row, s_col, _, _ = subpixel._correlate(ref, _fourier_shift(ref, dy, dx), WIN)
        assert abs(s_row / step - round(s_row / step)) < 1e-6
        assert abs(s_col / step - round(s_col / step)) < 1e-6


def test_larger_shifts_still_land_within_a_fifth_of_a_pixel():
    """Beyond ~1 px the Hann window eats the overlap; accuracy degrades but
    does not collapse, which is why there is a max-shift gate rather than a
    trust-everything policy."""
    ref = _band_limited(seed=3)
    for dy, dx in [(1.25, -2.5), (-2.0, 1.75)]:
        s_row, s_col, _, _ = subpixel._correlate(ref, _fourier_shift(ref, dy, dx), WIN)
        assert abs(s_row + dy) < 0.2
        assert abs(s_col + dx) < 0.2


def test_correlate_declines_a_flat_window():
    flat = np.full((WIN, WIN), 100.0, dtype=np.float32)
    assert subpixel._correlate(flat, flat.copy(), WIN) is None
    assert subpixel._correlate(_band_limited(), flat, WIN) is None


def test_response_separates_a_true_partner_from_a_mismatched_one():
    ref = _band_limited(seed=4)
    true_partner = _fourier_shift(ref, 0.3, -0.2)
    junk = _band_limited(seed=999)

    _, _, err_true, resp_true = subpixel._correlate(ref, true_partner, WIN)
    _, _, err_junk, resp_junk = subpixel._correlate(ref, junk, WIN)
    assert resp_true > 3 * resp_junk
    assert err_junk > err_true


# --------------------------------------------------------------------------
# the self-calibrating acceptance gate
# --------------------------------------------------------------------------

def _geoms(n, seed0=100, junk_idx=()):
    """``measured``-shaped entries: ``n`` window pairs, some deliberately wrong."""
    measured = []
    for i in range(n):
        ref = _band_limited(seed=seed0 + i)
        if i in junk_idx:
            mov = _band_limited(seed=50000 + i)  # unrelated content
        else:
            mov = _fourier_shift(ref, 0.25, -0.125)
        geom = {"c": i, "r": 2 * i, "win": WIN, "ref_g": ref, "mov_g": mov}
        measured.append((geom, subpixel._correlate(ref, mov, WIN)))
    return measured


def test_null_calibration_clears_true_partners_and_rejects_mismatched_ones():
    """The gate the paper calls self-calibrating: each window is scored against
    its own junk baseline, not an absolute floor."""
    junk_idx = {3, 11, 17}
    measured = _geoms(20, junk_idx=junk_idx)
    null_max = subpixel._null_max_responses(measured, workers=1)
    assert null_max is not None

    for geom, res in measured:
        floor = null_max[(geom["c"], geom["r"])]
        response = res[3]
        accepted = floor is not None and response > floor
        if geom["c"] in junk_idx:
            assert not accepted, f"window {geom['c']} is junk but passed the gate"
        else:
            assert accepted, f"window {geom['c']} is a true partner but was rejected"


def test_null_calibration_is_skipped_when_there_are_too_few_windows():
    """Below ``_NULL_MIN_SAMPLES`` there is nothing to calibrate against."""
    assert subpixel._null_max_responses(_geoms(subpixel._NULL_MIN_SAMPLES - 1),
                                        workers=1) is None
    assert subpixel._null_max_responses(_geoms(subpixel._NULL_MIN_SAMPLES),
                                        workers=1) is not None


def test_null_calibration_partner_draw_is_seeded():
    """The partner assignment is drawn from a fixed seed, not the global RNG."""
    n = 24
    draws = []
    for _ in range(3):
        rng = np.random.default_rng(1234)
        offsets = rng.integers(n // 4, max(n // 4 + 1, n - n // 4),
                               size=(n, subpixel._K_NULL))
        draws.append((np.arange(n)[:, None] + offsets) % n)
    assert np.array_equal(draws[0], draws[1])
    assert np.array_equal(draws[0], draws[2])
    # No partner is the window itself, so a null is always a content mismatch.
    assert not np.any(draws[0] == np.arange(n)[:, None])


def test_null_calibration_is_reproducible_when_run_sequentially():
    a = subpixel._null_max_responses(_geoms(subpixel._NULL_MIN_SAMPLES + 4),
                                     workers=1)
    b = subpixel._null_max_responses(_geoms(subpixel._NULL_MIN_SAMPLES + 4),
                                     workers=1)
    assert a == b


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT (not fixed here: this code produced the paper's "
           "numbers). cv2.phaseCorrelate multiplies BOTH inputs by the window "
           "in place whenever the array size is already an optimal DFT size "
           "(128 and 64 both are), so _correlate destroys the caller's "
           "ref_g/mov_g. _null_max_responses reuses those buffers across 8 "
           "partner correlations, so the k-th null sees Hann**k-scaled data "
           "and the result depends on evaluation order. Sequentially the order "
           "is fixed, so PATCH_WORKERS=1 -- the published setting -- is "
           "reproducible; with threads it is not. See test_determinism.py.")
def test_correlate_leaves_its_input_windows_unchanged():
    ref = _band_limited(seed=77)
    mov = _fourier_shift(ref, 0.25, -0.125)
    ref_before, mov_before = ref.copy(), mov.copy()
    subpixel._correlate(ref, mov, WIN)
    assert np.array_equal(ref, ref_before)
    assert np.array_equal(mov, mov_before)


def test_correlate_is_not_idempotent_because_of_that_defect():
    """Pins the observable consequence, so the defect cannot be lost silently.

    Delete this test together with the xfail above once the buffers are copied.
    """
    ref = _band_limited(seed=78)
    mov = _fourier_shift(ref, 0.25, -0.125)
    first = subpixel._correlate(ref, mov, WIN)
    second = subpixel._correlate(ref, mov, WIN)
    assert first != second


# --------------------------------------------------------------------------
# against a real raster pair
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def subpixel_pair(tmp_path_factory):
    """A pair whose true displacement is a fraction of a MOV pixel."""
    d = tmp_path_factory.mktemp("subpix")
    pair = _synth.make_pair(str(d), size=256, ref_gsd=GSD, mov_gsd=GSD,
                            d_fn=_synth.d_constant(0.4 * GSD, -0.3 * GSD),
                            name="subpix")
    ctx, st = init.run(pair.ref_path, pair.mov_path)
    return pair, ctx, st


def test_measure_point_recovers_a_subpixel_geographic_displacement(subpixel_pair):
    pair, ctx, st = subpixel_pair
    D = np.zeros_like(st.D)  # a zero prior: the measurement must find the shift
    out = subpixel.measure_point(ctx, D, 128, 128, subpixel._WIN_REFINE,
                                 subpixel._MAX_SHIFT_REFINE, 0.97)
    assert out is not None
    ddx, ddy, err, response = out
    assert ddx / GSD == pytest.approx(0.4, abs=0.15)
    assert ddy / GSD == pytest.approx(-0.3, abs=0.15)
    assert 0.0 <= response <= 1.0


def test_measure_point_declines_a_window_that_runs_off_the_grid(subpixel_pair):
    pair, ctx, st = subpixel_pair
    D = np.zeros_like(st.D)
    assert subpixel.measure_point(ctx, D, 2, 2, subpixel._WIN_REFINE,
                                  subpixel._MAX_SHIFT_REFINE, 0.97) is None


def test_measure_point_declines_when_the_shift_bound_is_exceeded(subpixel_pair):
    """A prior that is wrong by more than the bound must not be 'corrected'."""
    pair, ctx, st = subpixel_pair
    D = np.zeros_like(st.D)
    D[..., 0] = 60.0 * GSD  # 60 px away from the truth
    assert subpixel.measure_point(ctx, D, 128, 128, subpixel._WIN_REFINE,
                                  subpixel._MAX_SHIFT_REFINE, 0.97) is None


def test_measure_point_declines_when_the_error_gate_is_impossible(subpixel_pair):
    pair, ctx, st = subpixel_pair
    D = np.zeros_like(st.D)
    assert subpixel.measure_point(ctx, D, 128, 128, subpixel._WIN_REFINE,
                                  subpixel._MAX_SHIFT_REFINE, max_err=-1.0) is None


def test_refine_points_improves_a_deliberately_biased_field(subpixel_pair):
    """The stage's reason to exist: replace matcher noise with a measurement."""
    pair, ctx, st = subpixel_pair
    biased = st.copy()
    rng = np.random.default_rng(0)
    truth = np.array([0.4 * GSD, -0.3 * GSD])
    coords = [(c, r) for r in range(80, 180, 20) for c in range(80, 180, 20)]
    for c, r in coords:
        biased.F[r, c] = 1
        biased.C[r, c] = 0.5
        biased.D[r, c] = truth + rng.normal(0, 0.45 * GSD, 2)

    before = np.linalg.norm(biased.D[[r for _, r in coords],
                                     [c for c, _ in coords]] - truth, axis=1) / GSD
    n = subpixel.refine_points(ctx, biased, workers=1)
    assert n > 0
    after = np.linalg.norm(biased.D[[r for _, r in coords],
                                    [c for c, _ in coords]] - truth, axis=1) / GSD
    assert np.median(after) < np.median(before)
    assert np.median(after) < 0.2


def test_densify_grid_adds_lattice_points_where_there_were_none(subpixel_pair):
    pair, ctx, st = subpixel_pair
    work = st.copy()
    before = int(work.F.sum())
    added = subpixel.densify_grid(ctx, work, workers=1)
    assert added >= 0
    assert int(work.F.sum()) == before + added
    if added:
        truth = np.array([0.4 * GSD, -0.3 * GSD])
        ys, xs = np.where(work.F == 1)
        errs = np.linalg.norm(work.D[ys, xs] - truth, axis=1) / GSD
        assert np.median(errs) < 0.5


def test_point_confidence_is_clipped_to_a_usable_band():
    assert subpixel._point_conf(-5.0) == pytest.approx(0.05)
    assert subpixel._point_conf(0.5) == pytest.approx(0.5)
    assert subpixel._point_conf(20.0) == pytest.approx(0.95)


def test_stage_worker_count_defaults_to_the_configured_value(subpixel_pair,
                                                             monkeypatch):
    pair, ctx, st = subpixel_pair
    seen = []
    original = subpixel._run_batch

    def spy(ctx_, D, jobs, win, max_shift, err_q, workers):
        seen.append(workers)
        return original(ctx_, D, jobs, win, max_shift, err_q, workers)

    monkeypatch.setattr(subpixel, "_run_batch", spy)
    config.PATCH_WORKERS = 3
    subpixel.refine_points(ctx, st.copy())
    assert seen == [3]
