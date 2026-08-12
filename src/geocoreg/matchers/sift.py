import logging
import numpy as np
import cv2

from geocoreg.matchers.base import FeatureMatcher
from geocoreg.matchers.utils import apply_clahe

log = logging.getLogger("geocoreg.matchers.sift")

def _to_gray_u8(img):
    """Convert uint8 image to single-channel grayscale for feature extraction."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3:
        n = min(3, img.shape[2])
        return np.mean(img[:, :, :n].astype(np.float32), axis=2).astype(np.uint8)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")

def _rootsift(des):
    """Convert SIFT descriptors to RootSIFT (Hellinger kernel)."""
    des = des.copy()
    row_sums = des.sum(axis=1, keepdims=True) + 1e-7
    des /= row_sums
    np.sqrt(des, out=des)
    row_l2 = np.linalg.norm(des, axis=1, keepdims=True) + 1e-7
    des /= row_l2
    return des

def ransac_affine(pts_src, pts_dst, reproj_thresh):
    """Estimate affine transform with RANSAC."""
    if len(pts_src) < 3:
        return None, np.zeros(len(pts_src), dtype=bool), np.zeros(len(pts_src))

    M, inliers = cv2.estimateAffine2D(
        pts_src.astype(np.float64),
        pts_dst.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
    )

    if M is None or inliers is None:
        return None, np.zeros(len(pts_src), dtype=bool), np.zeros(len(pts_src))

    inlier_mask = inliers.ravel().astype(bool)

    # Compute reprojection errors for all points
    ones = np.ones((len(pts_src), 1), dtype=np.float64)
    src_h = np.hstack([pts_src.astype(np.float64), ones])  # (N, 3)
    projected = src_h @ M.T  # (N, 2)
    reproj_errors = np.sqrt(((projected - pts_dst.astype(np.float64))**2).sum(axis=1))

    return M, inlier_mask, reproj_errors

def sift_match(img_ref, img_mov, nfeatures, ratio_thresh, reproj_thresh):
    """Run SIFT matching and local RANSAC in one step."""
    img_ref_gray = apply_clahe(_to_gray_u8(img_ref))
    img_mov_gray = apply_clahe(_to_gray_u8(img_mov))

    sift = cv2.SIFT_create(nfeatures=nfeatures)

    kp_ref, des_ref = sift.detectAndCompute(img_ref_gray, None)
    kp_mov, des_mov = sift.detectAndCompute(img_mov_gray, None)

    if des_ref is None or des_mov is None or len(kp_ref) < 2 or len(kp_mov) < 2:
        return kp_ref, kp_mov, [], [], np.zeros(0, dtype=bool), np.zeros(0)

    # RootSIFT: L1-normalize, sqrt, then L2-normalize
    des_ref = _rootsift(des_ref)
    des_mov = _rootsift(des_mov)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_ref, des_mov, k=2)

    good_matches = []
    ratios = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        ratio = m.distance / n.distance if n.distance > 0 else 1.0
        if ratio < ratio_thresh:
            good_matches.append(m)
            ratios.append(ratio)

    if len(good_matches) < 4:
        inlier_mask = np.zeros(len(good_matches), dtype=bool)
        return kp_ref, kp_mov, good_matches, ratios, inlier_mask, np.zeros(len(good_matches))

    pts_ref = np.array([kp_ref[m.queryIdx].pt for m in good_matches])
    pts_mov = np.array([kp_mov[m.trainIdx].pt for m in good_matches])

    M, inlier_mask, reproj_errors = ransac_affine(pts_ref, pts_mov, reproj_thresh)
    if M is None:
        inlier_mask = np.zeros(len(good_matches), dtype=bool)
        reproj_errors = np.zeros(len(good_matches))
    if inlier_mask.sum() < 4:
        inlier_mask = np.zeros(len(good_matches), dtype=bool)

    return kp_ref, kp_mov, good_matches, ratios, inlier_mask, reproj_errors

def sift_match_multichannel(img_ref, img_mov, nfeatures, ratio_thresh, reproj_thresh, dedup_radius):
    """Run SIFT matching independently on each channel, merge, deduplicate, RANSAC."""
    if img_ref.ndim < 3 or img_mov.ndim < 3:
        return sift_match(img_ref, img_mov, nfeatures, ratio_thresh, reproj_thresh)

    n_ch_ref = img_ref.shape[2]
    n_ch_mov = img_mov.shape[2]
    if n_ch_ref != n_ch_mov or n_ch_ref == 1:
        return sift_match(img_ref, img_mov, nfeatures, ratio_thresh, reproj_thresh)

    n_channels = n_ch_ref
    sift = cv2.SIFT_create(nfeatures=nfeatures)

    all_pts_ref = []
    all_pts_mov = []
    all_ratios = []

    for ch in range(n_channels):
        ref_ch = apply_clahe(img_ref[:, :, ch])
        mov_ch = apply_clahe(img_mov[:, :, ch])

        kp_r, des_r = sift.detectAndCompute(ref_ch, None)
        kp_m, des_m = sift.detectAndCompute(mov_ch, None)

        if des_r is None or des_m is None or len(kp_r) < 2 or len(kp_m) < 2:
            continue

        des_r = _rootsift(des_r)
        des_m = _rootsift(des_m)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf.knnMatch(des_r, des_m, k=2)

        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            ratio = m.distance / n.distance if n.distance > 0 else 1.0
            if ratio < ratio_thresh:
                pt_r = kp_r[m.queryIdx].pt
                pt_m = kp_m[m.trainIdx].pt
                all_pts_ref.append(pt_r)
                all_pts_mov.append(pt_m)
                all_ratios.append(ratio)

    if len(all_pts_ref) < 4:
        return sift_match(img_ref, img_mov, nfeatures, ratio_thresh, reproj_thresh)

    pts_ref = np.array(all_pts_ref)
    pts_mov = np.array(all_pts_mov)
    ratios = all_ratios

    from scipy.spatial import cKDTree
    tree = cKDTree(pts_ref)
    pairs = tree.query_pairs(dedup_radius)

    suppress = set()
    for i, j in pairs:
        if i in suppress or j in suppress:
            continue
        if ratios[i] <= ratios[j]:
            suppress.add(j)
        else:
            suppress.add(i)

    keep = np.array([k for k in range(len(pts_ref)) if k not in suppress])
    if len(keep) < 4:
        return sift_match(img_ref, img_mov, nfeatures, ratio_thresh, reproj_thresh)

    pts_ref = pts_ref[keep]
    pts_mov = pts_mov[keep]
    ratios = [ratios[k] for k in keep]

    M, inlier_mask, reproj_errors = ransac_affine(pts_ref, pts_mov, reproj_thresh)
    if M is None:
        inlier_mask = np.zeros(len(pts_ref), dtype=bool)
        reproj_errors = np.zeros(len(pts_ref))
    if inlier_mask.sum() < 4:
        inlier_mask = np.zeros(len(pts_ref), dtype=bool)

    kp_ref_out = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1.0) for p in pts_ref]
    kp_mov_out = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1.0) for p in pts_mov]
    good_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=ratios[i]) for i in range(len(pts_ref))]

    log.debug(f"Multi-channel SIFT: {n_channels} channels, {len(all_pts_ref)} raw → {len(pts_ref)} deduped → {int(inlier_mask.sum())} inliers")

    return kp_ref_out, kp_mov_out, good_matches, ratios, inlier_mask, reproj_errors

class RootSIFTMatcher(FeatureMatcher):
    def __init__(self, nfeatures, ratio_thresh, reproj_thresh, multichan_dedup_radius):
        self.nfeatures = nfeatures
        self.ratio_thresh = ratio_thresh
        self.reproj_thresh = reproj_thresh
        self.multichan_dedup_radius = multichan_dedup_radius

    def match(self, img_ref: np.ndarray, img_mov: np.ndarray) -> tuple:
        multichan = (img_ref.ndim == 3 and img_mov.ndim == 3
                     and img_ref.shape[2] == img_mov.shape[2]
                     and img_ref.shape[2] > 1)
        if multichan:
            return sift_match_multichannel(
                img_ref, img_mov,
                self.nfeatures, self.ratio_thresh,
                self.reproj_thresh, self.multichan_dedup_radius
            )
        else:
            return sift_match(
                img_ref, img_mov,
                self.nfeatures, self.ratio_thresh,
                self.reproj_thresh
            )

    def match_backward(self, img_ref: np.ndarray, img_mov: np.ndarray) -> tuple:
        """Reverse match from moving to reference."""
        return self.match(img_mov, img_ref)
