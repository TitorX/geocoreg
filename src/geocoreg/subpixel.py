"""Subpixel refinement and densification via local phase correlation.

Both stages measure a translation between a MOV window and the REF window
predicted by the current displacement field, on the coarser of the two pixel
grids, using upsampled cross-correlation (matrix-multiply DFT — precision
~1/upsample of a coarse pixel):

* ``refine_points`` re-measures every matched point (F==1). Feature matchers
  localize keypoints to ~0.3-0.5 px of the coarse grid; the correlation
  measurement replaces that noise with a subpixel estimate, which matters
  most on cross-GSD pairs, where one coarse pixel spans many fine ones.
* ``densify_grid`` adds *new* control points on a uniform coarse-grid lattice
  seeded with the interpolated field prior — a second, complementary matcher
  that fills areas where the feature matcher found nothing.

Acceptance gating is self-calibrating per pair — there are no absolute
quality thresholds:

* response gate — each window pair's normalized phase-correlation peak power
  must beat the *maximum* of K correlations of the same REF window against
  far-away MOV windows (guaranteed content mismatches). This per-window null
  conditions on the window's own content: a smooth window is held to its own
  (high) junk standard, a textured window to its (low) one.
* err gate — correlation NRMS tracks radiometric similarity, not lock
  quality (cross-sensor pairs sit near 0.9 even for good locks), so the gate
  is a quantile of the pair's own err distribution.

Every measured point's confidence becomes its lock quality (one comparable
scale for all point sources); measurable points that *fail* the gates keep
their feature-match displacement but are downweighted, not pruned — pruning
deletes exactly the most informative anchors at displacement discontinuities.

Correspondence bookkeeping is done in pixel-center convention with the exact
resize scale factors, and converted to geographic displacement the same way
``refine`` does, so the two stages are drop-in producers of D/F/C updates.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from geocoreg import config, geo_utils
from geocoreg._sync import REF_READ_LOCK
from geocoreg.matchers.utils import robust_8bit_normalize

log = logging.getLogger("geocoreg.subpixel")

# Measurement internals. Windows are sized on the coarse grid; the refine
# window is larger (better SNR at feature-verified positions), the densify
# window smaller so window-average ≈ point value on nonrigid fields. Shift
# bounds cap the deviation from the prior (densify tighter: lattice points
# have no feature evidence). Err-gate quantiles cap per-pair acceptance.
_WIN_REFINE = 128
_WIN_DENSIFY = 64
_UPSAMPLE = 32            # DFT upsample factor (precision ~1/factor px)
_MAX_SHIFT_REFINE = 4.0   # coarse px
_MAX_SHIFT_DENSIFY = 3.0
_ERR_FLOOR = 0.6          # err gate is max(_ERR_FLOOR, quantile(errs, q))
_ERR_Q_REFINE = 0.5
_ERR_Q_DENSIFY = 0.4
_MIN_VALID_FRAC = 0.7     # min fraction of nonzero pixels per window
_DENSIFY_SPACING = 48     # lattice spacing on the coarse grid (px)
_K_NULL = 8               # null partners per window
_NULL_MIN_SAMPLES = 16    # below this, the response gate is skipped


def _to_gray_f32(img_u8):
    if img_u8.ndim == 3:
        n = min(3, img_u8.shape[2])
        return np.mean(img_u8[:, :, :n].astype(np.float32), axis=2)
    return img_u8.astype(np.float32)


def _hann(size):
    h = np.hanning(size).astype(np.float32)
    return np.outer(h, h)


def _extract_pair(ctx, D, c, r, win):
    """Extract co-located REF/MOV windows around overlap pixel (c, r).

    Uses ``D[r, c]`` as the prior locating the REF window. Both windows are
    normalized to gray float32 on the *work* (coarser) grid, ``win``x``win``.
    Returns a geometry dict for ``_finalize``, or None if the window is
    unusable (out of bounds, too much nodata, flat).
    """
    ref, mov = ctx.ref, ctx.mov
    H_ov, W_ov = D.shape[:2]

    # Local GSD ratio (REF px per MOV px), same convention as refine.
    mov_col_full = ctx.mov_crop[0] + c
    mov_row_full = ctx.mov_crop[1] + r
    ratio_x, ratio_y = geo_utils.compute_gsd_ratio(
        mov.gt, ref.gt, mov_col_full, mov_row_full, ctx.ct_mov2ref
    )
    ratio = max(ratio_x, ratio_y)
    if not np.isfinite(ratio) or ratio <= 0:
        return None

    # Window sizes on each native grid so both cover the same ground extent
    # and land on the coarser grid after resize.
    if ratio >= 1.0:      # REF finer -> work on MOV grid
        wm, wn = win, max(8, int(round(win * ratio)))
    else:                 # MOV finer -> work on REF grid
        wm, wn = max(8, int(round(win / ratio))), win
    fm = wm / win         # MOV native px per work px
    fr = wn / win         # REF native px per work px

    # MOV window (from the in-RAM overlap crop), fully inside or skip.
    c0m, r0m = c - wm // 2, r - wm // 2
    if c0m < 0 or r0m < 0 or c0m + wm > W_ov or r0m + wm > H_ov:
        return None
    mov_win = ctx.mov_overlap_data[r0m:r0m + wm, c0m:c0m + wm]

    # Predicted REF center from the displacement prior (as refine Step 3).
    g_mov_x, g_mov_y = geo_utils.pix2geo(ctx.gt_mov_overlap, c, r)
    dx, dy = D[r, c]
    g_sx, g_sy = g_mov_x + dx, g_mov_y + dy
    r_geo_x, r_geo_y = geo_utils.transform_point(ctx.ct_mov2ref, g_sx, g_sy)
    r_col, r_row = geo_utils.geo2pix(ref.gt, r_geo_x, r_geo_y)

    rc0 = int(round(r_col)) - wn // 2
    rr0 = int(round(r_row)) - wn // 2
    if rc0 < 0 or rr0 < 0 or rc0 + wn > ref.w or rr0 + wn > ref.h:
        return None
    with REF_READ_LOCK:
        ref_win = ref.ds.ReadAsArray(rc0, rr0, wn, wn)
    if ref_win is None:
        return None
    if ref_win.ndim == 3:
        ref_win = np.transpose(ref_win, (1, 2, 0))

    # Normalize to u8 (percentile stretch), gray, then bring the finer window
    # down to the work grid.
    mov_g = _to_gray_f32(robust_8bit_normalize(mov_win))
    ref_g = _to_gray_f32(robust_8bit_normalize(ref_win))
    if (mov_g > 0).mean() < _MIN_VALID_FRAC or (ref_g > 0).mean() < _MIN_VALID_FRAC:
        return None
    if wm != win:
        mov_g = cv2.resize(mov_g, (win, win), interpolation=cv2.INTER_AREA)
    if wn != win:
        ref_g = cv2.resize(ref_g, (win, win), interpolation=cv2.INTER_AREA)
    if ref_g.std() < 1e-3 or mov_g.std() < 1e-3:
        return None

    return {
        "c": c, "r": r, "win": win, "ref_g": ref_g, "mov_g": mov_g,
        "fm": fm, "fr": fr, "c0m": c0m, "r0m": r0m, "rc0": rc0, "rr0": rr0,
    }


def _correlate(ref_g, mov_g, win):
    """Upsampled phase correlation between two work-grid windows.

    Returns ``(s_row, s_col, err, response)`` or None. ``err`` is the NRMS
    of the upsampled correlation (radiometric-similarity scale); ``response``
    is the normalized phase-correlation peak power (lock-quality scale).
    """
    from skimage.registration import phase_cross_correlation

    hann = _hann(win)
    ref_w = (ref_g - ref_g.mean()) * hann
    mov_w = (mov_g - mov_g.mean()) * hann
    if ref_w.std() < 1e-3 or mov_w.std() < 1e-3:
        return None
    try:
        shift, err, _ = phase_cross_correlation(
            ref_w, mov_w, upsample_factor=_UPSAMPLE, normalization=None,
        )
        _, response = cv2.phaseCorrelate(ref_g, mov_g, hann)
    except Exception:
        return None
    err = float(err)
    response = float(response)
    if not (np.isfinite(err) and np.isfinite(response)):
        return None
    return float(shift[0]), float(shift[1]), err, response


def _finalize(ctx, geom, s_row, s_col):
    """Convert a work-grid shift into a geographic displacement (MOV CRS).

    Exact correspondence in pixel-center convention. Work-grid array index u
    maps to native index off + (u + 0.5) * f - 0.5; the array centers of the
    two windows correspond after applying the measured shift on the work
    grid (true REF index = center + shift, given skimage's sign convention
    verified as: shift = -(displacement of mov content relative to ref)).
    """
    win, fm, fr = geom["win"], geom["fm"], geom["fr"]
    pc = (win - 1) / 2.0
    mov_col_ov = geom["c0m"] + (pc + 0.5) * fm - 0.5
    mov_row_ov = geom["r0m"] + (pc + 0.5) * fm - 0.5
    ref_col_nat = geom["rc0"] + (pc + s_col + 0.5) * fr - 0.5
    ref_row_nat = geom["rr0"] + (pc + s_row + 0.5) * fr - 0.5

    mgx, mgy = geo_utils.pix2geo(
        ctx.mov.gt,
        mov_col_ov + ctx.mov_crop[0] + 0.5,
        mov_row_ov + ctx.mov_crop[1] + 0.5,
    )
    rgx, rgy = geo_utils.pix2geo(ctx.ref.gt, ref_col_nat + 0.5, ref_row_nat + 0.5)
    rmx, rmy = geo_utils.transform_point(ctx.ct_ref2mov, rgx, rgy)
    return float(rmx - mgx), float(rmy - mgy)


def measure_point(ctx, D, c, r, win, max_shift, max_err):
    """Single-point measurement (extract + correlate + gates + finalize).

    Returns ``(dx_geo, dy_geo, err, response)`` — the *corrected* displacement
    for this point in MOV CRS units — or ``None`` if the window is unusable
    or the correlation fails the gates.
    """
    geom = _extract_pair(ctx, D, c, r, win)
    if geom is None:
        return None
    res = _correlate(geom["ref_g"], geom["mov_g"], win)
    if res is None:
        return None
    s_row, s_col, err, response = res
    if err > max_err:
        return None
    if abs(s_row) > max_shift or abs(s_col) > max_shift:
        return None
    ddx, ddy = _finalize(ctx, geom, s_row, s_col)
    return ddx, ddy, err, response


def _null_max_responses(measured, workers):
    """Per-window junk-response reference.

    For each measured window pair, correlates the REF window against
    ``_K_NULL`` far-away MOV windows (row-major index distance > n/4 —
    guaranteed content mismatch) and returns the max null response per
    point, keyed by (c, r). Points whose nulls are all uncomputable map to
    None (conservatively unacceptable). Returns None when there are too few
    windows to calibrate at all.
    """
    n = len(measured)
    if n < _NULL_MIN_SAMPLES:
        return None
    rng = np.random.default_rng(1234)
    offsets = rng.integers(n // 4, max(n // 4 + 1, n - n // 4),
                           size=(n, _K_NULL))
    partners = (np.arange(n)[:, None] + offsets) % n

    def _one(i):
        g = measured[i][0]
        vals = []
        for j in partners[i]:
            m = _correlate(g["ref_g"], measured[j][0]["mov_g"], g["win"])
            if m is not None:
                vals.append(m[3])
        return max(vals) if vals else None

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            nmax = list(ex.map(_one, range(n)))
    else:
        nmax = [_one(i) for i in range(n)]
    valid = [v for v in nmax if v is not None]
    if valid:
        log.info(f"Null calibration: {n} windows x {_K_NULL} partners, "
                 f"null-max med={np.median(valid):.3f} "
                 f"q90={np.quantile(valid, 0.9):.3f}")
    return {(m[0]["c"], m[0]["r"]): v for m, v in zip(measured, nmax)}


def _run_batch(ctx, D, jobs, win, max_shift, err_q, workers):
    """measure over ``jobs`` [(c, r), ...] with per-pair self-calibration.

    Returns ``(accepted, rejected)``: accepted maps (c, r) to
    ``(ddx, ddy, err, response)``; rejected maps points that *were*
    measurable but failed a quality gate to their response (junk suspects).

    Gates: err below ``max(_ERR_FLOOR, quantile(errs, err_q))`` (capped at
    0.97 — absolute NRMS thresholds do not transfer across pair types),
    response above the per-window null max (``_null_max_responses``), and
    shift within ``max_shift`` of the prior.
    """
    def _ext(cr):
        c, r = cr
        try:
            return _extract_pair(ctx, D, c, r, win)
        except Exception as e:  # pragma: no cover — keep one failure local
            log.debug(f"extract({c},{r}) failed: {e}")
            return None

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            geoms = list(ex.map(_ext, jobs))
    else:
        geoms = [_ext(j) for j in jobs]
    geoms = [g for g in geoms if g is not None]

    def _meas(geom):
        res = _correlate(geom["ref_g"], geom["mov_g"], geom["win"])
        if res is None:
            return None
        return geom, res

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            measured = list(ex.map(_meas, geoms))
    else:
        measured = [_meas(g) for g in geoms]
    measured = [m for m in measured if m is not None]

    null_max = _null_max_responses(measured, workers)

    thr = _ERR_FLOOR
    if measured:
        errs = np.array([res[2] for _, res in measured])
        thr = min(0.97, max(_ERR_FLOOR, float(np.quantile(errs, err_q))))
        log.info(f"Adaptive err gate: thr={thr:.3f} "
                 f"(q={err_q}, err med={np.median(errs):.3f})")

    accepted, rejected = {}, {}
    for geom, (s_row, s_col, err, response) in measured:
        cr = (geom["c"], geom["r"])
        if null_max is not None:
            floor_i = null_max.get(cr)
            resp_ok = floor_i is not None and response > floor_i
        else:
            resp_ok = True
        if (err <= thr and resp_ok
                and abs(s_row) <= max_shift and abs(s_col) <= max_shift):
            ddx, ddy = _finalize(ctx, geom, s_row, s_col)
            accepted[cr] = (ddx, ddy, err, response)
        else:
            rejected[cr] = response
    return accepted, rejected


def _point_conf(response):
    """Confidence of a correlation-measured point (lock-quality scale)."""
    return float(np.clip(response, 0.05, 0.95))


def refine_points(ctx, state, workers=None):
    """Re-measure every matched point with phase correlation (in place).

    Points whose correlation passes the gates get their displacement replaced
    by the correlation estimate and their confidence set to the measured lock
    quality. Measurable points that fail the gates — junk-match suspects,
    typically low-texture areas producing mutually-consistent false matches
    that survive the consistency filters — keep their displacement but have
    their confidence scaled down by response relative to the accepted
    points' median, so they lose interpolation influence wherever better
    anchors exist. Returns the number refined.
    """
    t0 = time.time()
    workers = workers or max(1, config.PATCH_WORKERS)
    pts = np.argwhere(state.F == 1)
    jobs = [(int(c), int(r)) for r, c in pts]
    got, rejected = _run_batch(ctx, state.D, jobs, _WIN_REFINE,
                               _MAX_SHIFT_REFINE, _ERR_Q_REFINE, workers)
    for (c, r), (ddx, ddy, err, resp) in got.items():
        state.D[r, c, 0] = ddx
        state.D[r, c, 1] = ddy
        state.C[r, c] = _point_conf(resp)
    n_down = 0
    if rejected:
        resps = np.array(list(rejected.values()))
        good = np.array([res[3] for res in got.values()]) if got else resps
        scale_ref = float(np.quantile(good, 0.5)) if len(good) else 1.0
        for (c, r), resp in rejected.items():
            fac = np.clip(resp / max(scale_ref, 1e-6), 0.1, 1.0)
            state.C[r, c] = float(state.C[r, c] * fac)
            n_down += 1
    log.info(f"Subpixel refine: {len(got)}/{len(jobs)} points refined, "
             f"{n_down} downweighted ({time.time() - t0:.1f}s, "
             f"workers={workers})")
    return len(got)


def densify_grid(ctx, state, workers=None):
    """Add phase-correlation control points on a uniform lattice (in place).

    Lattice spacing is ``_DENSIFY_SPACING`` on the *coarse* grid, converted
    to MOV overlap pixels. Each candidate is seeded with the MLS-interpolated
    prior; only correlations passing the (tighter) densify gates become new
    points. Returns the number of points added.
    """
    from geocoreg.interp import fill as interp_fill

    t0 = time.time()
    workers = workers or max(1, config.PATCH_WORKERS)
    H, W = state.D.shape[:2]

    # Coarse-grid spacing -> MOV overlap px at the overlap center.
    ratio_x, ratio_y = geo_utils.compute_gsd_ratio(
        ctx.mov.gt, ctx.ref.gt,
        ctx.mov_crop[0] + W // 2, ctx.mov_crop[1] + H // 2, ctx.ct_mov2ref,
    )
    ratio = max(ratio_x, ratio_y)
    step = max(8, int(round(_DENSIFY_SPACING / min(1.0, ratio))))
    win = _WIN_DENSIFY

    # Half-size of the MOV-native window, to clamp lattice centers inward so
    # border areas still get measured (a window centered at the clamped
    # position fits; the measurement is assigned at that shifted center).
    half_m = (win if ratio >= 1.0 else int(round(win / ratio))) // 2 + 1
    if 2 * half_m >= min(H, W):
        return 0
    seen = set()
    centers = []
    for r in range(step // 2, H, step):
        for c in range(step // 2, W, step):
            cc = int(np.clip(c, half_m, W - 1 - half_m))
            rr = int(np.clip(r, half_m, H - 1 - half_m))
            if (cc, rr) not in seen and not state.F[rr, cc]:
                seen.add((cc, rr))
                centers.append((cc, rr))
    if not centers:
        return 0

    D_prior = interp_fill(state.D, state.F, targets=centers, C=state.C)
    got, _ = _run_batch(ctx, D_prior, centers, win,
                        _MAX_SHIFT_DENSIFY, _ERR_Q_DENSIFY, workers)
    for (c, r), (ddx, ddy, err, resp) in got.items():
        state.D[r, c, 0] = ddx
        state.D[r, c, 1] = ddy
        state.F[r, c] = 1
        state.C[r, c] = _point_conf(resp)
    log.info(f"Densify: {len(got)}/{len(centers)} lattice points added "
             f"(step={step} mov px, {time.time() - t0:.1f}s, workers={workers})")
    return len(got)
