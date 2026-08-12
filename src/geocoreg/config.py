"""Tunable parameters, as module-level constants.

Consumers read these as attributes at call time, so a setting can be changed
either here or by assigning the module attribute before calling the pipeline
(``import geocoreg.config as cfg; cfg.SUBPIX_REFINE = False``). They are
process-global. The defaults below are the ones all published results were
produced with; the method needs no per-pair tuning.
"""

# Phase 1 — Refine (multi-scale patch matching)

# Patches larger than this (longest side, pixels) are downsampled before
# feature detection, bounding matcher cost on the coarsest pyramid levels.
MAX_DOWNSAMPLE_SIDE = 2048
# Max SIFT keypoints detected per patch.
SIFT_NFEATURES = 2000
# Pyramid floor: levels halve the patch size until both sides fall below
# this, so smaller values mean more (finer) levels.
MIN_PATCH_SIZE = 256
# Lowe ratio test for descriptor matching; lower is stricter.
RATIO_THRESH = 0.75
# REF search window size relative to the MOV patch (1.0 = same footprint).
# Raise it when the initial misalignment approaches a patch width.
SEARCH_MARGIN_FACTOR = 1.0
# Skip a patch whose footprint already holds at least this many matched
# points, so later levels spend their time on the still-empty areas.
SPARSE_THRESHOLD = 5
# RANSAC reprojection threshold (pixels) for the per-patch affine gate; also
# the scale of the reprojection-error term in a point's confidence.
RANSAC_REPROJ_PATCH = 3.0
MULTICHAN_DEDUP_RADIUS = 3.0  # Suppress duplicate keypoints within this pixel radius
FWD_BWD_CONSISTENCY_PX = 3.0  # Max cycle error (pixels) for forward-backward check

# NMS (non-maximum suppression) on matched points after refinement: minimum
# spacing, in pixels, between kept anchors. Recall-first — keep more anchors
# and let the measurement gates + the local-affine filter reject, rather than
# thinning by fiat.
NMS_RADIUS = 32

# Outlier rejection. The per-level chain is: global magnitude MAD filter,
# then a leave-one-out local-affine residual filter (see geocoreg/filters.py —
# gradient-aware, threshold self-calibrated from the pair's own residuals).
MAD_K = 3.0  # MAD-based outlier threshold (k * 1.4826 * MAD from median)

# Interpolation (level priors and gap filling; the *final* fill is
# autotuned per pair, see INTERP_AUTOTUNE)
INTERP_K = 20          # neighboring control points per MLS estimate
INTERP_ALPHA = 2.0     # MLS inverse-distance weight exponent
INTERP_CHUNK_SIZE = 10000  # target pixels per batch; memory/speed only

# Parallelism inside refine (per-patch matching / correlation threads).
# 1 = sequential, the setting every published result was produced with.
# Raising it cuts wall-clock but does not always give an identical field:
# cv2.phaseCorrelate multiplies its inputs by the window function in place and
# subpixel.py reuses those buffers, so the null-response calibration depends on
# the order windows are visited in. Keep this at 1 when the field matters.
PATCH_WORKERS = 1

# Phase-correlation measurement stages (geocoreg/subpixel.py). Both are
# self-calibrating per pair — acceptance compares each window's correlation
# response against that window's own null (mismatched-partner) responses,
# plus an err gate at a fixed quantile of the pair's err distribution.
# Window sizes / upsampling / shift bounds are internal constants there.
SUBPIX_REFINE = True   # re-measure every matched point at subpixel precision
DENSIFY = True         # add lattice control points where features are absent

# Final fill: pick the interpolator per pair by 5-fold cross-validation over
# the pair's own control points (MLS k/alpha family, IDW, thin-plate spline).
# When False, the final fill is plain MLS at INTERP_K / INTERP_ALPHA.
INTERP_AUTOTUNE = True

# Debug: set to a path to save the final control points (pre-fill) as .npz.
DUMP_POINTS_PATH = None

# Phase 2 — Warp (GDAL geolocation arrays)
WARP_CHUNK_ROWS = 256  # rows per geolocation-array write; memory only
WARP_INTERP_METHOD = "nearest"  # Options: nearest, linear, cubic
