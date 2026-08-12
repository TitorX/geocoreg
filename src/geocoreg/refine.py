"""Multi-Scale Iterative Patch Refinement."""

import logging
import os
import time
import numpy as np
import cv2

from geocoreg import config, geo_utils, filters, viz
from geocoreg.matchers.utils import robust_8bit_normalize
from geocoreg.interp import fill as interp_fill

log = logging.getLogger("geocoreg.refine")


def _generate_levels(H, W):
    """Auto-generate refinement levels.

    Start at (H, W), halve each dimension until both < MIN_PATCH_SIZE.
    Returns list of (patch_h, patch_w) tuples, coarsest first.
    """
    levels = []
    ph, pw = H, W
    while ph >= config.MIN_PATCH_SIZE or pw >= config.MIN_PATCH_SIZE:
        levels.append((ph, pw))
        ph //= 2
        pw //= 2
    return levels


def run(ctx, state, matcher, diag_folder=None):
    """Execute refinement: unified multi-scale patch matching.

    Auto-generates levels from full MOV overlap size down to MIN_PATCH_SIZE,
    halving each dimension per level. Single pass per level.

    Args:
        ctx: InitContext
        state: DisplacementState (from Phase 0 or previous runs)
        matcher: FeatureMatcher implementation
        diag_folder: optional path to diagnostic folder
        
    Returns:
        New DisplacementState containing refined displacement field.
    """
    t0 = time.time()
    state = state.copy()  # Avoid mutating input state
    H, W = state.D.shape[:2]

    levels = _generate_levels(H, W)
    log.info(f"Auto-generated {len(levels)} levels: "
             + ", ".join(f"{pw}×{ph}" for ph, pw in levels))

    for lvl_idx, (patch_h, patch_w) in enumerate(levels, 1):
        spacing_r = patch_h
        spacing_c = patch_w

        log.info("-" * 60)
        log.info(f"Level {lvl_idx}/{len(levels)}: "
                 f"patch={patch_w}×{patch_h}, spacing={spacing_c}×{spacing_r}")

        centers = _uniform_grid(W, H, spacing_c, spacing_r)

        # Interpolate D at centers before matching
        state.D = interp_fill(state.D, state.F, targets=centers, C=state.C)

        # Determine which centers to skip (already dense)
        skip_count = 0
        active_centers = []
        for c, r in centers:
            half_c = patch_w // 2
            half_r = patch_h // 2
            r0 = max(0, r - half_r)
            r1 = min(H, r + half_r)
            c0 = max(0, c - half_c)
            c1 = min(W, c + half_c)
            local_count = int(state.F[r0:r1, c0:c1].sum())
            if local_count >= config.SPARSE_THRESHOLD:
                skip_count += 1
            else:
                active_centers.append((c, r))

        log.info(f"  Centers: {len(centers)} total, {skip_count} skipped (dense), "
                 f"{len(active_centers)} to match")

        if not active_centers:
            log.info(f"  All centers dense — skipping level")
            continue

        # Patch matching (sequential by default; threaded when PATCH_WORKERS>1 —
        # updates are merged by max confidence afterwards, so order is irrelevant)
        patches_matched = 0
        all_updates = []
        patch_bboxes = []  # (ref_bbox, mov_bbox) for matched patches only

        def _match_one(seq_center):
            patch_seq, (c, r) = seq_center
            try:
                return _match_patch_collect(
                    ctx, state, matcher, c, r, patch_w, patch_h,
                    lvl_idx, diag_folder, patch_seq,
                )
            except Exception as e:
                log.warning(f"Patch matching failed: {e}")
                return [], None, None, None

        workers = max(1, int(getattr(config, "PATCH_WORKERS", 1) or 1))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_match_one, enumerate(active_centers, start=1)))
        else:
            results = map(_match_one, enumerate(active_centers, start=1))

        for updates, ref_bbox, mov_bbox, patch_id in results:
            if updates:
                patches_matched += 1
                all_updates.extend(updates)
                if ref_bbox is not None:
                    patch_bboxes.append((ref_bbox, mov_bbox, patch_id))

        # Merge updates — keep best confidence per pixel
        best_updates = {}
        for rc, cc, dx, dy, conf in all_updates:
            key = (rc, cc)
            prev = best_updates.get(key)
            if prev is None or conf > prev[4]:
                best_updates[key] = (rc, cc, dx, dy, conf)

        new_matches = 0
        for rc, cc, dx, dy, conf in best_updates.values():
            if conf > state.C[rc, cc]:
                state.D[rc, cc, 0] = dx
                state.D[rc, cc, 1] = dy
                state.F[rc, cc] = 1
                state.C[rc, cc] = conf
                new_matches += 1

        if new_matches > 0:
            # Global MAD-based outlier filter
            filters.global_outlier_filter(state.D, state.F)
            # Leave-one-out local-affine residual filter (gradient-aware)
            filters.local_affine_filter(state.D, state.F, state.C)
            log.info(f"  Matched points after outlier filters: {state.F.sum()}")
            # NMS: for each region, keep only the highest-confidence matched point
            filters.nms_filter(state.D, state.F, state.C, config.NMS_RADIUS)
            log.info(f"  Matched points after NMS: {int(state.F.sum())}")

        total_d_points = int(state.F.sum())
        table_headers = [
            "MatchedPts",
            "MatchedPatches",
            "Attempted",
            "Skipped",
            "Total",
            "TotalDPoints",
        ]
        table_values = [
            str(new_matches),
            str(patches_matched),
            str(len(active_centers)),
            str(skip_count),
            str(len(centers)),
            str(total_d_points),
        ]
        col_widths = [max(len(h), len(v)) for h, v in zip(table_headers, table_values)]

        def _fmt_row(vals):
            return " | ".join(v.rjust(w) for v, w in zip(vals, col_widths))

        log.info("  Level Summary:")
        log.info(f"    {_fmt_row(table_headers)}")
        log.info("    " + "-+-".join("-" * w for w in col_widths))
        log.info(f"    {_fmt_row(table_values)}")
        log.info("    " + "-+-".join("-" * w for w in col_widths))

        # Per-level displacement field visualization
        if diag_folder:
            try:
                viz.viz_displacement_field_sparse(
                    ctx.mov_overlap_data, state.D, state.F, ctx.gt_mov_overlap,
                    os.path.join(diag_folder, f"refine_level{lvl_idx}_D.png"),
                    ref_img=ctx.ref_overlap_data,
                    patch_bboxes=patch_bboxes,
                )
            except Exception as e:
                log.warning(f"Per-level viz failed (level {lvl_idx}): {e}")

    log.info(f"Total matched points in D map: {int(state.F.sum())}")

    # Phase-correlation stages (see geocoreg.subpixel): re-measure the matched
    # points at subpixel precision, then add lattice points where the
    # feature matcher found nothing. Both feed the same filter chain.
    if config.SUBPIX_REFINE:
        from geocoreg import subpixel
        subpixel.refine_points(ctx, state)
    if config.DENSIFY:
        from geocoreg import subpixel
        if subpixel.densify_grid(ctx, state) > 0:
            filters.global_outlier_filter(state.D, state.F)
            filters.local_affine_filter(state.D, state.F, state.C)
        log.info(f"Points after densify + filters: {int(state.F.sum())}")

    # Debug: freeze the final control points (pre-fill) for offline
    # interpolation studies (see config.DUMP_POINTS_PATH).
    dump_path = getattr(config, "DUMP_POINTS_PATH", None)
    if dump_path:
        ys, xs = np.where(state.F == 1)
        np.savez(dump_path,
                 x=xs, y=ys,
                 d=state.D[ys, xs], c=state.C[ys, xs],
                 shape=np.array(state.D.shape[:2]),
                 mov_crop=np.array(ctx.mov_crop),
                 gt_mov_overlap=np.array(ctx.gt_mov_overlap))
        log.info(f"Dumped {len(xs)} control points to {dump_path}")

    # Full interpolation fill
    log.info("Interpolation fill of displacement field")
    if config.INTERP_AUTOTUNE:
        from geocoreg.interp import autotune_fill
        state.D = autotune_fill(state.D, state.F, C=state.C,
                                workers=max(1, config.PATCH_WORKERS))
    else:
        state.D = interp_fill(state.D, state.F, C=state.C)
    H, W = state.D.shape[:2]
    filled_mask = (state.F == 1) | (state.D[:, :, 0] != 0) | (state.D[:, :, 1] != 0)
    fill_pct = 100.0 * filled_mask.sum() / (H * W)
    log.info(f"Pixels filled: {fill_pct:.1f}%")

    dx_range = (state.D[:, :, 0].min(), state.D[:, :, 0].max())
    dy_range = (state.D[:, :, 1].min(), state.D[:, :, 1].max())
    mean_mag = np.sqrt(state.D[:, :, 0]**2 + state.D[:, :, 1]**2).mean()
    log.info(f"D range: dx=[{dx_range[0]:.2f}, {dx_range[1]:.2f}], "
             f"dy=[{dy_range[0]:.2f}, {dy_range[1]:.2f}] MOV CRS units")
    log.info(f"Mean |D|: {mean_mag:.2f} MOV CRS units")

    log.info(f"Refine complete ({time.time() - t0:.1f}s)")
    return state


def _uniform_grid(W, H, spacing_c, spacing_r):
    """Generate uniform grid centers."""
    centers = []
    for r in range(spacing_r // 2, H, spacing_r):
        for c in range(spacing_c // 2, W, spacing_c):
            centers.append((c, r))
    return centers


def _match_patch_collect(ctx, state, matcher, center_c, center_r, patch_w, patch_h,
                         level_idx, output_folder, patch_seq):
    """Match a single patch centered at (center_c, center_r) in MOV overlap coords.

    Returns:
        (updates, ref_bbox, mov_bbox, patch_id)
    """
    ref = ctx.ref
    mov = ctx.mov
    H_ov, W_ov = state.D.shape[:2]
    patch_id = f"p{patch_seq:05d}"

    # Step 1: Extract MOV patch (clip to edges) — this is the native patch
    half_w = patch_w // 2
    half_h = patch_h // 2
    c0_mov = max(0, center_c - half_w)
    c1_mov = min(W_ov, center_c + half_w)
    r0_mov = max(0, center_r - half_h)
    r1_mov = min(H_ov, center_r + half_h)

    mov_pw = c1_mov - c0_mov
    mov_ph = r1_mov - r0_mov
    if mov_pw < 16 or mov_ph < 16:
        return [], None, None, patch_id

    mov_patch = ctx.mov_overlap_data[r0_mov:r1_mov, c0_mov:c1_mov]

    # Step 2: Compute REF patch size using local GSD ratio (REF px per MOV px)
    mov_col_full = ctx.mov_crop[0] + center_c
    mov_row_full = ctx.mov_crop[1] + center_r
    ratio_x, ratio_y = geo_utils.compute_gsd_ratio(
        mov.gt, ref.gt, mov_col_full, mov_row_full, ctx.ct_mov2ref
    )
    ratio = max(ratio_x, ratio_y)

    ref_pw = max(32, int(round(mov_pw * ratio)))
    ref_ph = max(32, int(round(mov_ph * ratio)))

    # Step 3: Extract REF patch using current D prediction
    gt_ov = ctx.gt_mov_overlap
    g_mov_x, g_mov_y = geo_utils.pix2geo(gt_ov, center_c, center_r)
    dx, dy = state.D[center_r, center_c]
    g_sample_x = g_mov_x + dx
    g_sample_y = g_mov_y + dy

    r_geo_x, r_geo_y = geo_utils.transform_point(ctx.ct_mov2ref, g_sample_x, g_sample_y)
    r_col, r_row = geo_utils.geo2pix(ref.gt, r_geo_x, r_geo_y)

    margin = config.SEARCH_MARGIN_FACTOR
    ref_half_w = int(ref_pw * margin / 2)
    ref_half_h = int(ref_ph * margin / 2)
    rc0 = max(0, int(round(r_col)) - ref_half_w)
    rr0 = max(0, int(round(r_row)) - ref_half_h)
    rc1 = min(ref.w, int(round(r_col)) + ref_half_w)
    rr1 = min(ref.h, int(round(r_row)) + ref_half_h)

    if rc1 - rc0 < 16 or rr1 - rr0 < 16:
        return [], None, None, patch_id

    # GDAL dataset handles are not thread-safe — serialize reads (PATCH_WORKERS>1)
    from geocoreg._sync import REF_READ_LOCK
    with REF_READ_LOCK:
        ref_patch = ref.ds.ReadAsArray(rc0, rr0, rc1 - rc0, rr1 - rr0)
    if ref_patch is None:
        return [], None, None, patch_id
    if ref_patch.ndim == 2:
        ref_patch = ref_patch[:, :, np.newaxis]
    else:
        ref_patch = np.transpose(ref_patch, (1, 2, 0))

    # Bounding boxes in their respective overlap pixel coords
    mov_bbox = (c0_mov, r0_mov, c1_mov, r1_mov)
    rc0_ov = rc0 - ctx.ref_crop[0]
    rr0_ov = rr0 - ctx.ref_crop[1]
    rc1_ov = rc1 - ctx.ref_crop[0]
    rr1_ov = rr1 - ctx.ref_crop[1]
    ref_bbox = (rc0_ov, rr0_ov, rc1_ov, rr1_ov)

    # Step 4: Normalize to uint8 while preserving channel count
    ref_img_u8 = robust_8bit_normalize(ref_patch)
    mov_img_u8 = robust_8bit_normalize(mov_patch)

    # Step 5: Resolution normalization (GSD ratio, REF px per MOV px)
    if ratio > 1.5:
        # REF is finer than MOV — downsample REF patch
        target_w = int(ref_img_u8.shape[1] / ratio)
        target_h = int(ref_img_u8.shape[0] / ratio)
        if target_w > 0 and target_h > 0:
            ref_img_u8 = cv2.resize(ref_img_u8, (target_w, target_h),
                                    interpolation=cv2.INTER_AREA)
    elif ratio < 0.67:
        # MOV is finer than REF — downsample MOV patch
        target_w = int(mov_img_u8.shape[1] * ratio)
        target_h = int(mov_img_u8.shape[0] * ratio)
        if target_w > 0 and target_h > 0:
            mov_img_u8 = cv2.resize(mov_img_u8, (target_w, target_h),
                                    interpolation=cv2.INTER_AREA)

    # Step 6: MAX_DOWNSAMPLE_SIDE cap for both ref and mov
    max_side = config.MAX_DOWNSAMPLE_SIDE
    scale_ref = 1.0
    rh, rw = ref_img_u8.shape[:2]
    if max(rh, rw) > max_side:
        scale_ref = max_side / max(rh, rw)
        ref_img_u8 = cv2.resize(ref_img_u8,
                                (int(rw * scale_ref), int(rh * scale_ref)),
                                interpolation=cv2.INTER_AREA)

    scale_mov = 1.0
    mh, mw = mov_img_u8.shape[:2]
    if max(mh, mw) > max_side:
        scale_mov = max_side / max(mh, mw)
        mov_img_u8 = cv2.resize(mov_img_u8,
                                (int(mw * scale_mov), int(mh * scale_mov)),
                                interpolation=cv2.INTER_AREA)

    # Step 7/8: Match
    kp_ref, kp_mov, good_matches, ratios, inlier_mask, reproj_errors = matcher.match(ref_img_u8, mov_img_u8)

    pts_ref_local = np.array([kp_ref[m.queryIdx].pt for m in good_matches]) if good_matches else np.array([])
    pts_mov_local = np.array([kp_mov[m.trainIdx].pt for m in good_matches]) if good_matches else np.array([])

    # Save detailed per-patch match diagnostics into output folder.
    if output_folder:
        try:
            match_confidences = [1.0 - ratio for ratio in ratios]
            viz.viz_patch_match_figure(
                ref_img_u8,
                mov_img_u8,
                kp_ref,
                kp_mov,
                good_matches,
                out_dir=output_folder,
                level_idx=level_idx,
                patch_id=patch_id,
                confidences=match_confidences,
                inlier_mask=inlier_mask,
            )
            log.debug(f"Saved patch diagnostics id={patch_id} to {output_folder}")
        except Exception as e:
            log.warning(f"Failed to save patch diagnostics id={patch_id}: {e}")

    # Step 9: Validate local RANSAC result
    if len(inlier_mask) == 0 or inlier_mask.sum() == 0:
        return [], ref_bbox, mov_bbox, patch_id

    # Step 9b: Forward-backward consistency check
    bwd_kp_mov, bwd_kp_ref, bwd_matches, _, bwd_mask, _ = matcher.match_backward(ref_img_u8, mov_img_u8)

    fwd_inlier_idx = np.where(inlier_mask)[0]
    if len(bwd_mask) > 0 and bwd_mask.sum() > 0 and len(fwd_inlier_idx) > 0:
        from scipy.spatial import cKDTree

        # Reverse inlier pairs: bwd query=MOV, bwd train=REF
        bwd_inlier_idx = np.where(bwd_mask)[0]
        bwd_mov_pts = np.array([bwd_kp_mov[bwd_matches[i].queryIdx].pt for i in bwd_inlier_idx])
        bwd_ref_pts = np.array([bwd_kp_ref[bwd_matches[i].trainIdx].pt for i in bwd_inlier_idx])

        # Build kd-tree on reverse MOV-side points
        bwd_mov_tree = cKDTree(bwd_mov_pts)

        fwd_mov_pts = np.array([kp_mov[good_matches[i].trainIdx].pt for i in fwd_inlier_idx])
        fwd_ref_pts = np.array([kp_ref[good_matches[i].queryIdx].pt for i in fwd_inlier_idx])

        _, nn_idx = bwd_mov_tree.query(fwd_mov_pts)
        cycle_errors = np.linalg.norm(fwd_ref_pts - bwd_ref_pts[nn_idx], axis=1)

        cycle_ok = cycle_errors < config.FWD_BWD_CONSISTENCY_PX
        inlier_mask = np.zeros_like(inlier_mask)
        inlier_mask[fwd_inlier_idx[cycle_ok]] = True

    inlier_idx = np.where(inlier_mask)[0]
    if len(inlier_idx) == 0:
        return [], ref_bbox, mov_bbox, patch_id

    # Step 10: Convert to geographic coords and compute displacements.
    # Map keypoints from the resized matching images back to native patch
    # pixels. cv2.resize relates pixel centers by
    # src = (dst + 0.5) * (src_size / dst_size) - 0.5, and consecutive resizes
    # compose to the same form, so the exact inverse of the whole resize chain
    # (steps 5 and 6) only needs the native and final image sizes — nominal
    # factors (ratio/scale) would drift by the int() truncation of each resize.
    k_ref = np.array([(rc1 - rc0) / ref_img_u8.shape[1],
                      (rr1 - rr0) / ref_img_u8.shape[0]])
    k_mov = np.array([mov_pw / mov_img_u8.shape[1],
                      mov_ph / mov_img_u8.shape[0]])
    pts_ref_adj = (pts_ref_local[inlier_idx] + 0.5) * k_ref - 0.5
    pts_mov_scaled = (pts_mov_local[inlier_idx] + 0.5) * k_mov - 0.5

    # MOV local → MOV overlap → full-res MOV → MOV geo (original_mov_geo)
    pts_mov_overlap = pts_mov_scaled.copy()
    pts_mov_overlap[:, 0] += c0_mov
    pts_mov_overlap[:, 1] += r0_mov

    pts_mov_full = pts_mov_overlap.copy()
    pts_mov_full[:, 0] += ctx.mov_crop[0]
    pts_mov_full[:, 1] += ctx.mov_crop[1]

    pts_mov_geo = np.column_stack(geo_utils.pix2geo_array(
        mov.gt, pts_mov_full[:, 0] + 0.5, pts_mov_full[:, 1] + 0.5))

    # REF local → full-res REF → REF geo → MOV CRS (matched_ref_geo)
    pts_ref_full_col = pts_ref_adj[:, 0] + rc0
    pts_ref_full_row = pts_ref_adj[:, 1] + rr0

    pts_ref_geo = np.column_stack(geo_utils.pix2geo_array(
        ref.gt, pts_ref_full_col + 0.5, pts_ref_full_row + 0.5))

    ref_mov_x, ref_mov_y = geo_utils.transform_points(
        ctx.ct_ref2mov, pts_ref_geo[:, 0], pts_ref_geo[:, 1])

    updates = []
    for j in range(len(inlier_idx)):
        i_orig = inlier_idx[j]
        ddx = ref_mov_x[j] - pts_mov_geo[j, 0]
        ddy = ref_mov_y[j] - pts_mov_geo[j, 1]
        ratio_conf = 1.0 - ratios[i_orig] if i_orig < len(ratios) else 0.5
        reproj_conf = np.exp(-reproj_errors[i_orig] / config.RANSAC_REPROJ_PATCH)
        conf = ratio_conf * reproj_conf

        rc = int(round(pts_mov_overlap[j, 1]))
        cc = int(round(pts_mov_overlap[j, 0]))

        if 0 <= rc < H_ov and 0 <= cc < W_ov:
            updates.append((rc, cc, float(ddx), float(ddy), float(conf)))

    return updates, ref_bbox, mov_bbox, patch_id
