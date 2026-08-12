"""Pipeline orchestration — runs Phase 0 through Phase 2."""

import logging
import os
import time

import numpy as np

from geocoreg import init, refine, warp, control_points, viz, geo_utils, config
from geocoreg.matchers import RootSIFTMatcher

log = logging.getLogger("geocoreg")


def setup_logging(out_folder):
    """Configure logging: console (INFO) + file (DEBUG)."""
    logger = logging.getLogger("geocoreg")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-5s %(message)s"))
    logger.addHandler(ch)

    log_path = os.path.join(out_folder, "log.txt")
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))
    logger.addHandler(fh)

    return log_path


def run_match(ref_path, mov_path, out_folder):
    """Full pipeline: match + warp."""
    t0 = time.time()
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "matched.tif")
    setup_logging(out_folder)

    log.info("=== Dense Displacement Co-Registration Pipeline ===")
    log.info(f"REF: {ref_path}")
    log.info(f"MOV: {mov_path}")
    log.info(f"OUT: {out_path}")

    # Phase 0: Init
    log.info("=" * 60)
    log.info("Phase 0: Init")
    ctx, init_state = init.run(ref_path, mov_path)

    try:
        viz.viz_overlap(
            ctx.ref.read_full_downsampled(), ctx.mov.read_full_downsampled(),
            ctx.ref_crop, ctx.mov_crop,
            (ctx.ref.w, ctx.ref.h), (ctx.mov.w, ctx.mov.h),
            ctx.ref.crs_name, ctx.mov.crs_name,
            ctx.ref.filename, ctx.mov.filename,
            ctx.overlap_bbox, ctx.gsd_ratio_x, ctx.gsd_ratio_y,
            os.path.join(out_folder, "phase0_overlap.png"),
        )
    except Exception as e:
        log.warning(f"Phase 0 viz failed: {e}")

    # Phase 1: Refine (multi-scale patch matching)
    log.info("=" * 60)
    log.info("Phase 1: Refine (multi-scale patch matching)")
    
    matcher = RootSIFTMatcher(
        config.SIFT_NFEATURES,
        config.RATIO_THRESH,
        config.RANSAC_REPROJ_PATCH,
        config.MULTICHAN_DEDUP_RADIUS,
    )
    refined_state = refine.run(ctx, init_state, matcher, diag_folder=out_folder)

    # Save control points
    all_points = _extract_control_points(ctx, refined_state)
    control_points.write_control_points(
        os.path.join(out_folder, "control_points.geojson"),
        all_points, ctx.mov.crs_wkt, ref_path, mov_path,
    )
    control_points.write_points_file(
        os.path.join(out_folder, "control_points.points"),
        all_points, ctx.mov.crs_wkt,
    )

    try:
        viz.viz_displacement_field(
            ctx.mov_overlap_data, refined_state.D, ctx.gt_mov_overlap,
            os.path.join(out_folder, "refine_displacement_field.png"),
        )
    except Exception as e:
        log.warning(f"Displacement field viz failed: {e}")

    # Phase 2: Warp
    log.info("=" * 60)
    log.info("Phase 2: Warp")
    warp.run(ctx, refined_state, out_path)

    elapsed = time.time() - t0
    log.info(f"=== Pipeline complete in {elapsed:.1f}s ===")
    log.info(f"Output: {out_path}")


def run_apply(ref_path, mov_path, out_folder, cp_path):
    """Apply existing control points to warp."""
    t0 = time.time()
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "matched.tif")
    setup_logging(out_folder)

    log.info("=== Dense Displacement Co-Registration — Apply Mode ===")
    log.info(f"REF: {ref_path}")
    log.info(f"MOV: {mov_path}")
    log.info(f"OUT: {out_path}")
    log.info(f"CP:  {cp_path}")

    # Phase 0: Init
    ctx, init_state = init.run(ref_path, mov_path)

    # Load and rasterize control points
    log.info(f"Apply mode: loading control points from {cp_path}")
    points, cp_crs_wkt = control_points.read_control_points(cp_path)
    ct_cp2mov = geo_utils.make_transform(cp_crs_wkt, ctx.mov.crs_wkt) if cp_crs_wkt else None

    state = init_state.copy()

    D, F, C, n_inside = control_points.rasterize_control_points(
        points, ctx.gt_mov_overlap, state.D.shape[0], state.D.shape[1], ct_cp2mov
    )
    state.D, state.F, state.C = D, F, C

    log.info(f"Apply mode: {len(points)} points loaded, {n_inside} within current overlap "
             f"({len(points) - n_inside} outside)")
    log.info("Apply mode: skipping Phase 1, proceeding to interpolation + warp")

    # Interpolation fill — same selection rule as run_match, so applying a run's
    # own control points reproduces that run's field.
    if config.INTERP_AUTOTUNE:
        from geocoreg.interp import autotune_fill
        state.D = autotune_fill(state.D, state.F, C=state.C,
                                workers=max(1, config.PATCH_WORKERS))
    else:
        from geocoreg.interp import fill as interp_fill
        state.D = interp_fill(state.D, state.F, C=state.C)

    # Phase 2: Warp
    warp.run(ctx, state, out_path)

    elapsed = time.time() - t0
    log.info(f"=== Apply complete in {elapsed:.1f}s ===")
    log.info(f"Output: {out_path}")


def _extract_control_points(ctx, state):
    """Build control point list from matched pixels in D/F/C arrays."""
    H, W = state.F.shape
    fy, fx = np.where(state.F == 1)

    mov_full_col = fx.astype(np.float64) + ctx.mov_crop[0]
    mov_full_row = fy.astype(np.float64) + ctx.mov_crop[1]
    mov_geo_x, mov_geo_y = geo_utils.pix2geo_array(
        ctx.mov.gt, mov_full_col + 0.5, mov_full_row + 0.5)

    dx = state.D[fy, fx, 0]
    dy = state.D[fy, fx, 1]
    conf = state.C[fy, fx].astype(np.float64)
    corrected_x = mov_geo_x + dx
    corrected_y = mov_geo_y + dy

    points = [
        {
            "mov_geo_x": float(mov_geo_x[i]),
            "mov_geo_y": float(mov_geo_y[i]),
            "corrected_x": float(corrected_x[i]),
            "corrected_y": float(corrected_y[i]),
            "dx": float(dx[i]),
            "dy": float(dy[i]),
            "confidence": float(conf[i]),
        }
        for i in range(len(fy))
    ]

    log.info(f"Extracted {len(points)} control points from displacement map")
    return points
