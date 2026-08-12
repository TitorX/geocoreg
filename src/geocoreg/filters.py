"""Outlier filtering on displacement fields."""

import logging
import numpy as np

log = logging.getLogger("geocoreg.filters")


def local_affine_filter(D, F, C=None, k=12, mad_k=3.5):
    """Leave-one-out local-affine residual filter.

    For every matched point, predict its displacement from its k nearest
    matched neighbors (self excluded) with the same weighted-affine model MLS
    interpolation uses, and measure the residual. Points whose residual
    exceeds a robust global threshold (median + mad_k * 1.4826 * MAD of all
    residuals) are removed.

    Unlike a median-of-neighbors deviation test, this respects smooth
    displacement *gradients* — a point on a valid ramp fits its neighbors'
    affine model exactly, while a mismatched point does not. Unlike a
    windowed magnitude/angle MAD test, it has no arbitrary window geometry and no degenerate angle statistics when the
    true displacement is near zero. The threshold is self-calibrating: it is
    a scale of this pair's own residual distribution.
    """
    from scipy.spatial import cKDTree

    matched = np.argwhere(F == 1)
    n = len(matched)
    if n < k + 3:
        return 0
    rows = matched[:, 0]
    cols = matched[:, 1]
    coords = np.stack([cols, rows], axis=-1).astype(np.float64)
    vals = D[rows, cols, :].astype(np.float64)
    conf = (C[rows, cols].astype(np.float64)
            if C is not None else np.ones(n))

    tree = cKDTree(coords)
    kk = min(k + 1, n)
    dists, idx = tree.query(coords, k=kk)
    nbr_d = dists[:, 1:]                       # drop self (distance-0 column)
    nbr_i = idx[:, 1:]

    w_geom = 1.0 / (nbr_d ** 2 + 1e-8)
    w_geom = w_geom * np.clip(conf[nbr_i], 1e-3, None)

    def _fit(w):
        """Weighted local affine LOO prediction + studentized residuals."""
        w = w / w.sum(axis=1, keepdims=True)
        p = coords[nbr_i]                      # (n, k, 2) neighbor positions
        q = vals[nbr_i]                        # (n, k, 2) neighbor values
        w3 = w[:, :, np.newaxis]
        p_star = np.sum(w3 * p, axis=1, keepdims=True)
        q_star = np.sum(w3 * q, axis=1, keepdims=True)
        hp = p - p_star
        hq = q - q_star
        w4 = w[:, :, np.newaxis, np.newaxis]
        A = np.sum(w4 * (hp[:, :, :, np.newaxis] @ hp[:, :, np.newaxis, :]), axis=1)
        B = np.sum(w4 * (hp[:, :, :, np.newaxis] @ hq[:, :, np.newaxis, :]), axis=1)

        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        trace = A[:, 0, 0] + A[:, 1, 1]
        A_inv = np.empty_like(A)
        safe_det = np.where(np.abs(det) > 1e-12, det, 1.0)
        A_inv[:, 0, 0] = A[:, 1, 1] / safe_det
        A_inv[:, 1, 1] = A[:, 0, 0] / safe_det
        A_inv[:, 0, 1] = -A[:, 0, 1] / safe_det
        A_inv[:, 1, 0] = -A[:, 1, 0] / safe_det
        M = A_inv @ B
        v_c = coords[:, np.newaxis, :] - p_star        # (n, 1, 2)
        pred = (v_c @ M).squeeze(1) + q_star.squeeze(1)
        # Degenerate neighborhoods (collinear neighbors): fall back to the
        # weighted mean instead of an exploding affine extrapolation.
        degenerate = np.abs(det) <= 1e-9 * np.maximum(trace, 1.0) ** 2
        pred[degenerate] = q_star.squeeze(1)[degenerate]

        # Studentize: the prediction is linear in the neighbor values with
        # coefficients a_j = w_j (1 + (v-p*)^T A^-1 hp_j), so its noise
        # scales with g = sqrt(sum a_j^2). Extrapolating fits (hull edges,
        # sparse areas) have g >> interior points; without this a threshold
        # calibrated on the interior bulk cuts valid high-leverage points.
        s = ((v_c @ A_inv) @ np.transpose(hp, (0, 2, 1))).squeeze(1)  # (n, k)
        a = w * (1.0 + s)
        g2 = np.sum(a ** 2, axis=1)
        g2[degenerate] = np.sum(w[degenerate] ** 2, axis=1)
        return np.linalg.norm(vals - pred, axis=1) / np.sqrt(1.0 + g2)

    # IRLS: one reweighting pass. Outliers corrupt the fits of *neighboring*
    # points (and, at displacement discontinuities, the two sides corrupt
    # each other), inflating good points' residuals. Downweighting neighbors
    # by their own first-pass residual (Tukey biweight) cleans the fits, so
    # only genuinely inconsistent points stay above threshold.
    res = _fit(w_geom)
    scale = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
    u = res[nbr_i] / (4.685 * scale)
    tukey = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
    res = _fit(w_geom * (tukey + 1e-6))

    med = np.median(res)
    mad = np.median(np.abs(res - med))
    threshold = med + mad_k * 1.4826 * mad + 1e-9
    bad = res > threshold

    n_removed = int(bad.sum())
    if n_removed > 0:
        D[rows[bad], cols[bad], 0] = 0.0
        D[rows[bad], cols[bad], 1] = 0.0
        F[rows[bad], cols[bad]] = 0
        if C is not None:
            C[rows[bad], cols[bad]] = 0.0

    log.info(f"Local affine filter: removed {n_removed}/{n} points "
             f"(threshold={threshold:.4f}, k={kk - 1})")
    return n_removed

def nms_filter(D, F, C, radius):
    from scipy.spatial import cKDTree

    matched = np.argwhere(F == 1)
    if len(matched) < 2:
        return 0

    rows = matched[:, 0]
    cols = matched[:, 1]
    coords = np.stack([cols, rows], axis=-1).astype(np.float32)
    confs = C[rows, cols]

    order = np.argsort(-confs)
    suppress = np.zeros(len(matched), dtype=bool)

    tree = cKDTree(coords)
    for idx in order:
        if suppress[idx]:
            continue
        nbrs = tree.query_ball_point(coords[idx], r=radius)
        for j in nbrs:
            if j != idx:
                suppress[j] = True

    n_removed = int(suppress.sum())
    if n_removed > 0:
        D[rows[suppress], cols[suppress], 0] = 0.0
        D[rows[suppress], cols[suppress], 1] = 0.0
        F[rows[suppress], cols[suppress]] = 0

    log.debug(f"NMS filter: removed {n_removed} points "
             f"(radius={radius}, kept={len(matched) - n_removed})")
    return n_removed

def global_outlier_filter(D, F):
    from geocoreg import config

    matched = np.argwhere(F == 1)
    if len(matched) < 4:
        return 0

    rows = matched[:, 0]
    cols = matched[:, 1]
    dx = D[rows, cols, 0]
    dy = D[rows, cols, 1]
    magnitudes = np.sqrt(dx**2 + dy**2)

    median_mag = np.median(magnitudes)
    mad = np.median(np.abs(magnitudes - median_mag))
    threshold = median_mag + config.MAD_K * 1.4826 * mad

    bad = magnitudes > threshold

    n_removed = int(bad.sum())
    if n_removed > 0:
        D[rows[bad], cols[bad], 0] = 0.0
        D[rows[bad], cols[bad], 1] = 0.0
        F[rows[bad], cols[bad]] = 0

    log.info(f"Global outlier filter (MAD): removed {n_removed} points "
             f"(median={median_mag:.4f}, mad={mad:.4f}, threshold={threshold:.4f})")
    return n_removed
