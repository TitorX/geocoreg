"""Determinism and concurrency.

Repeated runs at the published setting (``PATCH_WORKERS = 1``) are identical.
Bit-identity *across* thread counts is not guaranteed: ``cv2.phaseCorrelate``
overwrites the arrays it is handed, so the null-response calibration in
``subpixel._null_max_responses`` depends on the order its correlations run in.
That over-claim is recorded as an xfail rather than fixed — this code produced
the paper's numbers.
"""

from __future__ import annotations

import threading

import _synth
import numpy as np
import pytest

from geocoreg import config, init, refine
from geocoreg.matchers import RootSIFTMatcher

# 384 px is the smallest size at which the phase-correlation stages collect
# enough windows (>= _NULL_MIN_SAMPLES) for the null calibration to run, which
# is where the order dependence lives.
SIZE = 384


@pytest.fixture(scope="module")
def worker_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("determinism")
    pair = _synth.make_pair(str(d), size=SIZE,
                            d_fn=_synth.d_constant(35.0, -25.0), name="det")
    ctx, st = init.run(pair.ref_path, pair.mov_path)
    return pair, ctx, st


def _refine_with(ctx, st, workers):
    config.PATCH_WORKERS = workers
    matcher = RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                              config.RANSAC_REPROJ_PATCH,
                              config.MULTICHAN_DEDUP_RADIUS)
    return refine.run(ctx, st, matcher, diag_folder=None)


@pytest.fixture(scope="module")
def sequential_baseline(worker_pair):
    _, ctx, st = worker_pair
    saved = config.PATCH_WORKERS
    try:
        out = _refine_with(ctx, st, 1)
    finally:
        config.PATCH_WORKERS = saved
    return out.D.copy(), out.F.copy()


# --------------------------------------------------------------------------
# repeatability
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_the_same_input_twice_gives_the_same_output(worker_pair):
    """No unseeded randomness anywhere in the default configuration."""
    _, ctx, st = worker_pair
    first = _refine_with(ctx, st, 1)
    second = _refine_with(ctx, st, 1)
    assert np.array_equal(first.D, second.D)
    assert np.array_equal(first.F, second.F)
    assert np.array_equal(first.C, second.C)


@pytest.mark.slow
def test_refine_does_not_mutate_the_state_it_was_given(worker_pair):
    _, ctx, st = worker_pair
    before = (st.D.copy(), st.F.copy(), st.C.copy())
    _refine_with(ctx, st, 1)
    assert np.array_equal(st.D, before[0])
    assert np.array_equal(st.F, before[1])
    assert np.array_equal(st.C, before[2])


# --------------------------------------------------------------------------
# PATCH_WORKERS
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("workers", [4, 8])
@pytest.mark.xfail(
    strict=False,
    reason="KNOWN OVER-CLAIM in config.py ('Results are identical for any "
           "value'). subpixel._correlate hands its window buffers straight to "
           "cv2.phaseCorrelate, which multiplies them by the Hann window in "
           "place; _null_max_responses then reuses those buffers across "
           "partner correlations, so the acceptance gate depends on the order "
           "the threads happen to run in. Intermittent by nature, hence "
           "strict=False: it depends on the scheduler. NOT fixed here -- this "
           "is the code the paper's numbers came from. The published setting "
           "PATCH_WORKERS=1 is unaffected and is covered above.")
def test_patch_workers_do_not_change_the_result(worker_pair, sequential_baseline,
                                                workers):
    _, ctx, st = worker_pair
    d_ref, f_ref = sequential_baseline
    out = _refine_with(ctx, st, workers)
    assert np.array_equal(out.D, d_ref)
    assert np.array_equal(out.F, f_ref)


@pytest.mark.slow
@pytest.mark.parametrize("workers", [4, 8])
def test_patch_workers_agree_to_well_within_a_pixel(worker_pair,
                                                    sequential_baseline, workers):
    """What is actually guaranteed: the thread count perturbs the acceptance
    gate, not the registration. Bounded here so a real regression — a matcher
    genuinely misbehaving under threads — would still be caught."""
    pair, ctx, st = worker_pair
    d_ref, _ = sequential_baseline
    out = _refine_with(ctx, st, workers)
    err = np.linalg.norm(out.D - d_ref, axis=-1) / pair.mov_gsd
    assert np.median(err) < 0.05
    assert err.max() < 0.5


@pytest.mark.slow
def test_a_threaded_run_is_still_internally_consistent(worker_pair):
    pair, ctx, st = worker_pair
    out = _refine_with(ctx, st, 8)
    assert out.D.shape == st.D.shape
    assert np.isfinite(out.D).all()
    truth = pair.gt_field(ctx.gt_mov_overlap, ctx.mov_crop[2], ctx.mov_crop[3])
    assert np.median(_synth.epe_pixels(out.D, truth, pair.mov_gsd)) < 1.0


# --------------------------------------------------------------------------
# the GDAL read lock
# --------------------------------------------------------------------------

def test_ref_read_lock_is_a_module_level_lock():
    from geocoreg import _sync

    assert isinstance(_sync.REF_READ_LOCK, type(threading.Lock()))
    # Every threaded reader must share the one lock instance.
    from geocoreg import subpixel

    assert subpixel.REF_READ_LOCK is _sync.REF_READ_LOCK


@pytest.mark.slow
def test_concurrent_locked_reads_never_return_the_wrong_block(worker_pair):
    """Stress the pattern every threaded stage uses to read REF.

    A GDAL dataset handle is not thread-safe; concurrent ``ReadAsArray`` calls
    on one handle can return another thread's block. Each read here is checked
    against the same window read single-threaded beforehand, so a swapped block
    fails loudly rather than showing up as a mysterious mismatch later.
    """
    from geocoreg._sync import REF_READ_LOCK

    _, ctx, _ = worker_pair
    ref = ctx.ref
    rng = np.random.default_rng(0)
    win = 32
    windows = [(int(rng.integers(0, ref.w - win)), int(rng.integers(0, ref.h - win)))
               for _ in range(160)]
    expected = {}
    for c0, r0 in windows:
        expected[(c0, r0)] = np.asarray(ref.ds.ReadAsArray(c0, r0, win, win))

    errors = []

    def worker(items):
        for c0, r0 in items:
            with REF_READ_LOCK:
                got = np.asarray(ref.ds.ReadAsArray(c0, r0, win, win))
            if not np.array_equal(got, expected[(c0, r0)]):
                errors.append((c0, r0))

    n_threads = 8
    threads = [threading.Thread(target=worker, args=(windows[i::n_threads],))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"{len(errors)} concurrent reads returned the wrong block"


@pytest.mark.slow
def test_interpolation_is_thread_count_independent(worker_pair):
    """The final fill is pure numpy/scipy, so it must not care about workers."""
    from geocoreg import interp

    rng = np.random.default_rng(0)
    D = np.zeros((96, 96, 2))
    F = np.zeros((96, 96), np.uint8)
    C = np.zeros((96, 96), np.float32)
    idx = rng.choice(96 * 96, size=300, replace=False)
    ys, xs = np.unravel_index(idx, (96, 96))
    D[ys, xs, 0] = 0.2 * xs - 0.1 * ys
    D[ys, xs, 1] = 0.05 * xs + 0.3 * ys
    F[ys, xs] = 1
    C[ys, xs] = 1.0

    a = interp.autotune_fill(D.copy(), F.copy(), C=C.copy(), workers=1)
    b = interp.autotune_fill(D.copy(), F.copy(), C=C.copy(), workers=8)
    assert np.array_equal(a, b)
