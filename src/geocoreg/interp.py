"""Interpolation of displacement field D — fills unfilled pixels."""

import logging
import time
import numpy as np
from scipy.spatial import cKDTree

from geocoreg import config

log = logging.getLogger("geocoreg.interp")


def fill(D, F, targets=None, C=None):
    """Fill unfilled pixels in D using Moving Least Squares interpolation.

    Matched pixels (F==1) are kept unchanged.  Unfilled pixels (F==0) are
    interpolated from the k nearest matched pixels using an MLS affine model
    with inverse-distance weighting.

    When C (confidence) is provided, weights are scaled by confidence so that
    low-confidence anchor points have less influence on the interpolated field.

    Args:
        D: (H, W, 2) displacement field
        F: (H, W) uint8, 1 = matched pixel
        targets: optional list of (col, row) tuples — if provided, only these
                 unfilled pixels are interpolated instead of all unfilled pixels.
        C: optional (H, W) float32 confidence array (0-1). If provided, MLS
           weights are multiplied by confidence values.

    Returns:
        D_filled: (H, W, 2) filled displacement field
    """
    t0 = time.time()
    H, W = D.shape[:2]
    D_out = D.copy()

    known_y, known_x = np.where(F == 1)
    if len(known_y) == 0:
        log.warning("No anchor points for interpolation fill")
        return D_out

    if targets is not None:
        # Only fill specified target pixels that are not already matched
        tgt = np.array(targets, dtype=np.int32)  # (N, 2) with col, row
        mask = F[tgt[:, 1], tgt[:, 0]] == 0
        tgt = tgt[mask]
        if len(tgt) == 0:
            return D_out
        unfilled_x = tgt[:, 0]
        unfilled_y = tgt[:, 1]
    else:
        unfilled_y, unfilled_x = np.where(F == 0)
        if len(unfilled_y) == 0:
            log.info("Interp fill: all pixels already filled")
            return D_out

    known_coords = np.stack([known_x, known_y], axis=-1).astype(np.float32)
    known_values = D[known_y, known_x, :]
    known_conf = C[known_y, known_x] if C is not None else None
    target_coords = np.stack([unfilled_x, unfilled_y], axis=-1).astype(np.float32)

    tree = cKDTree(known_coords)
    k = min(config.INTERP_K, len(known_coords))
    alpha = config.INTERP_ALPHA
    chunk_size = config.INTERP_CHUNK_SIZE

    num_unfilled = len(target_coords)
    interpolated = np.zeros((num_unfilled, 2), dtype=np.float32)

    for i in range(0, num_unfilled, chunk_size):
        end = min(i + chunk_size, num_unfilled)
        batch_coords = target_coords[i:end]

        dists, indices = tree.query(batch_coords, k=k)

        # Ensure 2-D even when k == 1
        if k == 1:
            dists = dists[:, np.newaxis]
            indices = indices[:, np.newaxis]

        weights = 1.0 / (np.power(dists, alpha) + 1e-8)
        if known_conf is not None:
            weights *= known_conf[indices]  # scale by anchor confidence
        weights /= weights.sum(axis=1, keepdims=True)

        p = known_coords[indices]          # (batch, k, 2)
        q = known_values[indices]          # (batch, k, 2)

        w = weights[:, :, np.newaxis]      # (batch, k, 1)
        p_star = np.sum(w * p, axis=1, keepdims=True)  # (batch, 1, 2)
        q_star = np.sum(w * q, axis=1, keepdims=True)  # (batch, 1, 2)

        hat_p = p - p_star                 # (batch, k, 2)
        hat_q = q - q_star                 # (batch, k, 2)

        w4 = weights[:, :, np.newaxis, np.newaxis]  # (batch, k, 1, 1)
        A = np.sum(w4 * (hat_p[:, :, :, np.newaxis] @ hat_p[:, :, np.newaxis, :]), axis=1)
        B = np.sum(w4 * (hat_p[:, :, :, np.newaxis] @ hat_q[:, :, np.newaxis, :]), axis=1)

        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0] + 1e-8
        A_inv = np.empty_like(A)
        A_inv[:, 0, 0] =  A[:, 1, 1] / det
        A_inv[:, 1, 1] =  A[:, 0, 0] / det
        A_inv[:, 0, 1] = -A[:, 0, 1] / det
        A_inv[:, 1, 0] = -A[:, 1, 0] / det

        M = A_inv @ B                     # (batch, 2, 2)
        v_centered = batch_coords[:, np.newaxis, :] - p_star  # (batch, 1, 2)
        batch_res = (v_centered @ M).squeeze(1) + q_star.squeeze(1)

        interpolated[i:end] = batch_res

    D_out[unfilled_y, unfilled_x, :] = interpolated

    elapsed = time.time() - t0
    log.info(f"Interp fill: {len(known_y)} known → {len(unfilled_y)} filled "
             f"(k={k}, {elapsed:.1f}s)")

    return D_out


# ── final-fill autotune: per-pair interpolator selection by hold-out CV ──────
# The best interpolator is pair-dependent: thin-plate splines suit
# dense/elastic point sets, low-alpha MLS sparse noisy ones, IDW cases where
# the affine term overfits. Hold-out CV over the pair's own control points
# picks among them — no ground truth and no user knob needed.

_CANDIDATES = (
    ("mls", 8, 2.0), ("mls", 12, 2.0), ("mls", 20, 1.0), ("mls", 20, 2.0),
    ("mls", 32, 2.0), ("idw", 20, 2.0), ("tps", 32, None),
)


def _fill_mls(D, F, C, k, alpha, targets=None):
    from geocoreg import config as _c
    old = (_c.INTERP_K, _c.INTERP_ALPHA)
    _c.INTERP_K, _c.INTERP_ALPHA = k, alpha
    try:
        return fill(D, F, targets=targets, C=C)
    finally:
        _c.INTERP_K, _c.INTERP_ALPHA = old


def _predict_idw(pts, vals, conf, targets, k, alpha):
    tree = cKDTree(pts)
    kk = min(k, len(pts))
    out = np.empty((len(targets), 2))
    for i in range(0, len(targets), 20000):
        t = targets[i:i + 20000]
        d, j = tree.query(t, k=kk)
        if kk == 1:
            d, j = d[:, None], j[:, None]
        w = 1.0 / (np.power(d, alpha) + 1e-8)
        if conf is not None:
            w = w * conf[j]
        w /= w.sum(axis=1, keepdims=True)
        out[i:i + 20000] = (w[..., None] * vals[j]).sum(axis=1)
    return out


def _predict_tps(pts, vals, targets, neighbors=32, smoothing=1.0, workers=1):
    from scipy.interpolate import RBFInterpolator
    r = RBFInterpolator(pts, vals, kernel="thin_plate_spline",
                        neighbors=min(neighbors, len(pts)),
                        smoothing=smoothing)
    chunks = [targets[i:i + 50000] for i in range(0, len(targets), 50000)]
    if workers > 1 and len(chunks) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(r, chunks))
    else:
        parts = [r(c) for c in chunks]
    return np.concatenate(parts, axis=0)


def _predict(name, k, alpha, pts, vals, conf, targets, H, W, workers=1):
    if name == "tps":
        return _predict_tps(pts, vals, targets, workers=workers)
    if name == "idw":
        return _predict_idw(pts, vals, conf, targets, k, alpha)
    D2 = np.zeros((H, W, 2), dtype=np.float64)
    F2 = np.zeros((H, W), dtype=np.uint8)
    C2 = np.zeros((H, W), dtype=np.float32)
    xs, ys = pts[:, 0].astype(int), pts[:, 1].astype(int)
    D2[ys, xs] = vals
    F2[ys, xs] = 1
    C2[ys, xs] = conf if conf is not None else 1.0
    out = _fill_mls(D2, F2, C2, k, alpha,
                    targets=[(int(c), int(r)) for c, r in targets])
    return out[targets[:, 1].astype(int), targets[:, 0].astype(int)]


def autotune_fill(D, F, C=None, workers=1, folds=5):
    """Full fill with the interpolator a cross-validation picks for this pair.

    K-fold CV (fixed seed) over the matched points: every point is predicted
    once from the other folds, every candidate is scored on all points —
    single-split CV on a few hundred points is too noisy to rank near-tied
    candidates. The winner refits all points and fills every unmatched pixel.
    Falls back to the default ``fill`` when there are too few points.
    """
    t0 = time.time()
    ys, xs = np.where(F == 1)
    n = len(ys)
    if n < 50:
        return fill(D, F, C=C)
    pts = np.stack([xs, ys], -1).astype(np.float64)
    vals = D[ys, xs, :].astype(np.float64)
    conf = C[ys, xs].astype(np.float64) if C is not None else None
    H, W = F.shape

    perm = np.random.default_rng(42).permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[perm] = np.arange(n) % folds
    scores = {}
    for name, k, alpha in _CANDIDATES:
        try:
            errs = np.empty(n)
            for f_i in range(folds):
                test = np.where(fold_of == f_i)[0]
                train = np.where(fold_of != f_i)[0]
                pred = _predict(name, k, alpha, pts[train], vals[train],
                                conf[train] if conf is not None else None,
                                pts[test], H, W, workers=workers)
                errs[test] = np.linalg.norm(pred - vals[test], axis=1)
            scores[(name, k, alpha)] = float(errs.mean())
        except Exception as e:  # pragma: no cover — skip broken candidate
            log.warning(f"autotune candidate {name} k={k} a={alpha} failed: {e}")
    if not scores:
        return fill(D, F, C=C)
    (name, k, alpha), best = min(scores.items(), key=lambda kv: kv[1])
    cur = scores.get(("mls", 20, 2.0), float("nan"))
    log.info("Interp autotune: " + "  ".join(
        f"{n_}k{k_}a{a_ if a_ else ''}={v:.4f}"
        for (n_, k_, a_), v in sorted(scores.items(), key=lambda kv: kv[1])))
    log.info(f"Interp autotune winner: {name} k={k} alpha={alpha} "
             f"(cv {best:.4f} vs default {cur:.4f}, n={n})")

    if name == "mls":
        return _fill_mls(D, F, C, k, alpha)
    uy, ux = np.where(F == 0)
    targets = np.stack([ux, uy], -1).astype(np.float64)
    pred = _predict(name, k, alpha, pts, vals, conf, targets, H, W,
                    workers=workers)
    D_out = D.copy()
    D_out[uy, ux, :] = pred
    log.info(f"Interp autotune fill: {n} known → {len(uy)} filled "
             f"({time.time() - t0:.1f}s)")
    return D_out
