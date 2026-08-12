"""All diagnostic visualizations."""

import logging
import os
import numpy as np
import cv2

# matplotlib is an optional dependency: the pipeline runs fine without it and
# only the figure-producing helpers below need it.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    _HAVE_MATPLOTLIB = True
except ImportError:  # pragma: no cover - environment-dependent
    matplotlib = None
    plt = None
    Rectangle = None
    _HAVE_MATPLOTLIB = False

log = logging.getLogger("geocoreg.viz")


def _require_matplotlib():
    if not _HAVE_MATPLOTLIB:
        raise RuntimeError(
            "Visualization requires matplotlib; install the optional extra: "
            "pip install -e '.[viz]'"
        )


def _geo_displacement_to_pixel(gt, dx_geo, dy_geo):
    """Convert CRS displacement vectors into image pixel displacement vectors."""
    det = gt[1] * gt[5] - gt[2] * gt[4]
    dx_pix = (gt[5] * dx_geo - gt[2] * dy_geo) / det
    dy_pix = (-gt[4] * dx_geo + gt[1] * dy_geo) / det
    return dx_pix, dy_pix


def _prep_raster_for_display(img, max_side=2000):
    """Normalize a raster to [0, 1] float for display, then downsample."""
    if img is None or img.size == 0:
        return np.zeros((100, 100), dtype=np.float32)

    from geocoreg.matchers.utils import robust_8bit_normalize

    disp = robust_8bit_normalize(img).astype(np.float32) / 255.0

    h, w = disp.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        import cv2
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    return disp


def viz_patch_match_figure(
    ref_img_u8,
    mov_img_u8,
    kp_ref,
    kp_mov,
    matches,
    out_dir,
    level_idx,
    patch_id,
    confidences=None,
    inlier_mask=None,
):
    """Save per-patch match visualization in the workflow output folder."""
    patch_dir = os.path.join(out_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    fig_path = os.path.join(
        patch_dir, f"patch_level{level_idx}_{patch_id}_matches.png")

    def _to_vis_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] >= 3:
            return img[:, :, :3].copy()
        if img.ndim == 3 and img.shape[2] == 2:
            h, w, _ = img.shape
            z = np.zeros((h, w, 1), dtype=img.dtype)
            return np.concatenate([img, z], axis=2)
        raise ValueError(f"Unsupported image shape for visualization: {img.shape}")

    ref_vis = _to_vis_bgr(ref_img_u8)
    mov_vis = _to_vis_bgr(mov_img_u8)

    h1, w1 = ref_vis.shape[:2]
    h2, w2 = mov_vis.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    fig = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    fig[:h1, :w1] = ref_vis
    fig[:h2, w1:w1 + w2] = mov_vis

    if confidences is None:
        confidences = [None] * len(matches)

    if inlier_mask is None:
        draw_pairs = list(zip(matches, confidences))
    else:
        draw_pairs = [
            (m, conf)
            for m, conf, keep in zip(matches, confidences, inlier_mask.tolist())
            if int(keep) == 1
        ]

    n_pairs = len(draw_pairs)
    for i, (m, conf) in enumerate(draw_pairs):
        if n_pairs > 1:
            hue_idx = int(round(255.0 * i / (n_pairs - 1)))
        else:
            hue_idx = 0
        color = cv2.applyColorMap(
            np.array([[hue_idx]], dtype=np.uint8),
            cv2.COLORMAP_HSV,
        )[0, 0]
        color_bgr = (int(color[0]), int(color[1]), int(color[2]))

        p_ref = kp_ref[m.queryIdx].pt
        p_mov = kp_mov[m.trainIdx].pt

        x1, y1 = int(round(p_ref[0])), int(round(p_ref[1]))
        x2, y2 = int(round(p_mov[0] + w1)), int(round(p_mov[1]))

        cv2.circle(fig, (x1, y1), 3, color_bgr, 1, lineType=cv2.LINE_AA)
        cv2.circle(fig, (x2, y2), 3, color_bgr, 1, lineType=cv2.LINE_AA)
        cv2.line(fig, (x1, y1), (x2, y2), color_bgr, 1, lineType=cv2.LINE_AA)

        if conf is not None:
            conf_label = f"{conf:.2f}"
            cv2.putText(fig, conf_label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_bgr,
                        1, cv2.LINE_AA)
            cv2.putText(fig, conf_label, (x2 + 5, y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color_bgr,
                        1, cv2.LINE_AA)

    cv2.putText(fig, "REF", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(fig, "MOV", (w1 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(fig_path, fig)
    log.debug(f"Saved: {fig_path}")

    return patch_id


def viz_overlap(ref_img, mov_img, ref_crop, mov_crop, ref_dims, mov_dims,
                ref_crs, mov_crs, ref_file, mov_file, overlap_bbox,
                gsd_ratio_x, gsd_ratio_y, save_path):
    """Phase 0: overlap extent visualization."""
    _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ref_disp = _prep_raster_for_display(ref_img)
    ax1.imshow(ref_disp, cmap='gray', aspect='equal')
    rh, rw = ref_disp.shape[:2]
    scale_r = rh / ref_dims[1]
    rc = ref_crop
    ax1.add_patch(Rectangle(
        (rc[0] * scale_r, rc[1] * scale_r),
        rc[2] * scale_r, rc[3] * scale_r,
        linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.2
    ))
    ax1.set_title(f"REF: {ref_file}\n{ref_dims[0]}×{ref_dims[1]} px, {ref_crs}", fontsize=9)
    ax1.axis('off')

    mov_disp = _prep_raster_for_display(mov_img)
    ax2.imshow(mov_disp, cmap='gray', aspect='equal')
    mh, mw = mov_disp.shape[:2]
    scale_m = mh / mov_dims[1]
    mc = mov_crop
    ax2.add_patch(Rectangle(
        (mc[0] * scale_m, mc[1] * scale_m),
        mc[2] * scale_m, mc[3] * scale_m,
        linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.2
    ))
    ax2.set_title(f"MOV: {mov_file}\n{mov_dims[0]}×{mov_dims[1]} px, {mov_crs}", fontsize=9)
    ax2.axis('off')

    fig.suptitle(
        f"Overlap: x=[{overlap_bbox[0]:.1f}, {overlap_bbox[1]:.1f}] "
        f"y=[{overlap_bbox[2]:.1f}, {overlap_bbox[3]:.1f}]\n"
        f"GSD ratio: 1 REF px ≈ {gsd_ratio_x:.1f} MOV px (x), {gsd_ratio_y:.1f} MOV px (y)",
        fontsize=9
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.debug(f"Saved: {save_path}")


def viz_displacement_field(img, D, gt, save_path):
    """Displacement field visualization (quiver + HSV map).

    Args:
        img: background raster the displacement field is defined on (MOV).
        D: (H, W, 2) displacement field, native to img's grid.
        gt: geotransform of img/D, used to convert D_geo into pixel vectors.
    """
    _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    img_disp = _prep_raster_for_display(img)
    ax1.imshow(img_disp, cmap='gray', aspect='equal')

    H, W = D.shape[:2]
    step_r = max(H // 50, 1)
    step_c = max(W // 50, 1)
    rr = np.arange(0, H, step_r)
    cc = np.arange(0, W, step_c)
    CC, RR = np.meshgrid(cc, rr)

    dh, dw = img_disp.shape[:2]
    scale_y = dh / H
    scale_x = dw / W

    dx_sub, dy_sub = _geo_displacement_to_pixel(
        gt, D[RR, CC, 0], D[RR, CC, 1])
    mags = np.sqrt(dx_sub**2 + dy_sub**2)

    q = ax1.quiver(
        CC * scale_x, RR * scale_y,
        dx_sub * scale_x, dy_sub * scale_y,
        mags, cmap='hot', scale_units='xy', angles='xy', alpha=0.7
    )
    plt.colorbar(q, ax=ax1, label='Displacement magnitude (pixels)', shrink=0.7)
    ax1.set_title("Displacement Field (Quiver)", fontsize=11)
    ax1.axis('off')

    # HSV displacement map: hue=direction, value=magnitude
    dx_full, dy_full = _geo_displacement_to_pixel(gt, D[:, :, 0], D[:, :, 1])
    mag_full = np.sqrt(dx_full**2 + dy_full**2)
    angle = np.arctan2(dy_full, dx_full)

    hue = (angle + np.pi) / (2.0 * np.pi)
    sat = np.where(mag_full > 0, 1.0, 0.0)
    mag_scale = np.percentile(mag_full, 95)
    if mag_scale <= 1e-10:
        mag_scale = 1.0
    val = np.clip(mag_full / mag_scale, 0.0, 1.0)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    ax2.imshow(rgb, aspect='equal')
    ax2.set_title("Displacement Field (HSV: hue=angle, value=magnitude)", fontsize=11)
    ax2.axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.debug(f"Saved: {save_path}")


def viz_displacement_field_sparse(mov_img, D, F, gt_mov, save_path,
                                  max_arrows=5000,
                                  ref_img=None, patch_bboxes=None):
    """Per-level displacement field quiver at matched points.

    Left panel: MOV overlap (native grid of D/F) with displacement quiver and
    matched patch boundaries.
    Right panel (when ref_img + patch_bboxes provided): REF overlap with the
    corresponding REF patch boundaries.

    Each matched patch pair shares the same random color so pairs are visually
    linked across panels.
    """
    _require_matplotlib()
    two_panel = ref_img is not None and patch_bboxes

    if two_panel:
        fig, (ax_mov, ax_ref) = plt.subplots(1, 2, figsize=(22, 10))
    else:
        fig, ax_mov = plt.subplots(figsize=(12, 10))
        ax_ref = None

    mov_disp = _prep_raster_for_display(mov_img)
    ax_mov.imshow(mov_disp, cmap='gray', aspect='equal')

    H, W = D.shape[:2]
    dh, dw = mov_disp.shape[:2]
    scale_y = dh / H
    scale_x = dw / W

    if patch_bboxes:
        rng = np.random.default_rng(42)
        colors = np.clip(rng.random((len(patch_bboxes), 3)) * 0.85 + 0.15, 0, 1)

        if two_panel:
            ref_disp = _prep_raster_for_display(ref_img)
            ax_ref.imshow(ref_disp, cmap='gray', aspect='equal')
            orig_ref_h, orig_ref_w = ref_img.shape[:2]
            dh_ref, dw_ref = ref_disp.shape[:2]
            scale_ref_y = dh_ref / orig_ref_h
            scale_ref_x = dw_ref / orig_ref_w

        for i, patch in enumerate(patch_bboxes):
            if len(patch) >= 3:
                ref_bbox, mov_bbox, patch_id = patch[0], patch[1], patch[2]
            else:
                ref_bbox, mov_bbox = patch
                patch_id = None
            color = colors[i]
            c0m, r0m, c1m, r1m = mov_bbox
            mov_center_x = (c0m + c1m) / 2 * scale_x
            mov_center_y = (r0m + r1m) / 2 * scale_y

            # MOV patch rectangle (fill + border)
            for facecolor, alpha in [(color, 0.15), ('none', 0.7)]:
                ax_mov.add_patch(Rectangle(
                    (c0m * scale_x, r0m * scale_y),
                    (c1m - c0m) * scale_x, (r1m - r0m) * scale_y,
                    linewidth=1.0, edgecolor=color, facecolor=facecolor, alpha=alpha,
                ))
            ax_mov.plot(mov_center_x, mov_center_y,
                        '+', color=color, markersize=5, markeredgewidth=0.8, alpha=0.9)
            if patch_id:
                ax_mov.text(mov_center_x, mov_center_y, str(patch_id),
                            color='white', fontsize=6, ha='center', va='center')

            if two_panel and ref_bbox is not None:
                c0r, r0r, c1r, r1r = ref_bbox
                ref_center_x = (c0r + c1r) / 2 * scale_ref_x
                ref_center_y = (r0r + r1r) / 2 * scale_ref_y
                for facecolor, alpha in [(color, 0.15), ('none', 0.7)]:
                    ax_ref.add_patch(Rectangle(
                        (c0r * scale_ref_x, r0r * scale_ref_y),
                        (c1r - c0r) * scale_ref_x, (r1r - r0r) * scale_ref_y,
                        linewidth=1.0, edgecolor=color, facecolor=facecolor, alpha=alpha,
                    ))
                ax_ref.plot(ref_center_x, ref_center_y,
                            '+', color=color, markersize=5, markeredgewidth=0.8, alpha=0.9)
                if patch_id:
                    ax_ref.text(ref_center_x, ref_center_y, str(patch_id),
                                color='white', fontsize=6, ha='center', va='center')

        if two_panel:
            ax_ref.set_title(
                f"REF (overlap) — {len(patch_bboxes)} matched patches", fontsize=11)
            ax_ref.axis('off')

    # Quiver of displacement vectors on MOV panel
    fy, fx = np.where(F == 1)
    n_matched = len(fy)
    if n_matched == 0:
        mov_title = "MOV — no matched points"
    else:
        if len(fy) > max_arrows:
            idx = np.random.default_rng(0).choice(len(fy), max_arrows, replace=False)
            fy, fx = fy[idx], fx[idx]

        dx_q, dy_q = _geo_displacement_to_pixel(
            gt_mov, D[fy, fx, 0], D[fy, fx, 1])
        mags = np.sqrt(dx_q**2 + dy_q**2)

        q = ax_mov.quiver(
            fx * scale_x, fy * scale_y,
            dx_q * scale_x, dy_q * scale_y,
            mags, cmap='berlin', scale_units='xy', angles='xy', alpha=0.7,
        )
        plt.colorbar(q, ax=ax_mov, label='Displacement magnitude (pixels)', shrink=0.7)
        n_patches_str = (f", {len(patch_bboxes)} matched patches"
                         if patch_bboxes else "")
        mov_title = f"MOV — {n_matched} matched points{n_patches_str}"

    ax_mov.set_title(mov_title, fontsize=11)
    ax_mov.axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.debug(f"Saved: {save_path}")
