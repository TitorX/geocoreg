"""The ``FeatureMatcher`` plug point.

``RootSIFTMatcher`` must satisfy the six-tuple contract in
``matchers/base.py`` down to the index semantics — ``refine.py`` dereferences
``queryIdx`` into the *reference* keypoints and ``trainIdx`` into the *moving*
ones, and swapping them silently mirrors every displacement. The second half
checks that the contract alone is enough for an outside matcher to drop in.
"""

from __future__ import annotations

import _synth
import cv2
import numpy as np
import pytest

from geocoreg import config
from geocoreg.matchers import FeatureMatcher, RootSIFTMatcher

SHIFT_X, SHIFT_Y = 9, -6
PATCH = 176


@pytest.fixture(scope="module")
def shifted_patches():
    """``(img_ref, img_mov)`` where a feature at ref ``p`` sits at mov
    ``p - (SHIFT_X, SHIFT_Y)`` — an integer crop offset, so the correspondence
    is exact rather than interpolated."""
    pad = 32
    base = _synth.texture(PATCH + 2 * pad, PATCH + 2 * pad, seed=42)
    base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    img_ref = base[pad:pad + PATCH, pad:pad + PATCH]
    img_mov = base[pad + SHIFT_Y:pad + SHIFT_Y + PATCH,
                   pad + SHIFT_X:pad + SHIFT_X + PATCH]
    return np.ascontiguousarray(img_ref), np.ascontiguousarray(img_mov)


@pytest.fixture(scope="module")
def matcher():
    return RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                           config.RANSAC_REPROJ_PATCH,
                           config.MULTICHAN_DEDUP_RADIUS)


@pytest.fixture(scope="module")
def matched(matcher, shifted_patches):
    """One forward match, reused: SIFT is the expensive part of this module."""
    return matcher.match(*shifted_patches)


# --------------------------------------------------------------------------
# the six-tuple contract
# --------------------------------------------------------------------------

def test_match_returns_the_documented_six_tuple(matched):
    result = matched
    assert isinstance(result, tuple)
    assert len(result) == 6

    kp_ref, kp_mov, matches, ratios, inlier_mask, reproj_errors = result
    assert all(isinstance(k, cv2.KeyPoint) for k in kp_ref[:5])
    assert all(isinstance(k, cv2.KeyPoint) for k in kp_mov[:5])
    assert all(isinstance(m, cv2.DMatch) for m in matches[:5])

    n = len(matches)
    assert n > 0, "the synthetic patches must produce matches"
    assert len(ratios) == n
    assert len(inlier_mask) == n
    assert len(reproj_errors) == n

    assert np.asarray(inlier_mask).dtype == np.bool_
    assert np.asarray(reproj_errors).dtype.kind == "f"
    assert np.all(np.asarray(reproj_errors) >= 0.0)


def test_match_indices_stay_inside_their_keypoint_lists(matched):
    kp_ref, kp_mov, matches, _, _, _ = matched
    for m in matches:
        assert 0 <= m.queryIdx < len(kp_ref)
        assert 0 <= m.trainIdx < len(kp_mov)


def test_ratios_respect_the_lowe_threshold_and_lower_is_better(matched):
    _, _, _, ratios, _, _ = matched
    ratios = np.asarray(ratios, dtype=float)
    assert np.all(ratios >= 0.0)
    assert np.all(ratios < config.RATIO_THRESH), "a match above the threshold survived"


def test_a_looser_ratio_threshold_admits_more_matches(shifted_patches):
    """Pins the direction of the ratio test: it is a rejection threshold."""
    strict = RootSIFTMatcher(config.SIFT_NFEATURES, 0.6,
                             config.RANSAC_REPROJ_PATCH,
                             config.MULTICHAN_DEDUP_RADIUS)
    loose = RootSIFTMatcher(config.SIFT_NFEATURES, 0.9,
                            config.RANSAC_REPROJ_PATCH,
                            config.MULTICHAN_DEDUP_RADIUS)
    n_strict = len(strict.match(*shifted_patches)[2])
    n_loose = len(loose.match(*shifted_patches)[2])
    assert n_loose > n_strict


def test_query_indexes_reference_and_train_indexes_moving(matched):
    """The known crop offset must come back with the right sign.

    ``kp_ref[queryIdx] - kp_mov[trainIdx]`` is the reference-minus-moving pixel
    offset; getting the two indices the wrong way round negates it.
    """
    kp_ref, kp_mov, matches, _, inlier_mask, _ = matched
    idx = np.where(np.asarray(inlier_mask))[0]
    assert len(idx) >= 4, "RANSAC found no consistent set on a pure translation"

    d = np.array([np.subtract(kp_ref[matches[i].queryIdx].pt,
                              kp_mov[matches[i].trainIdx].pt) for i in idx])
    assert np.median(d[:, 0]) == pytest.approx(SHIFT_X, abs=0.75)
    assert np.median(d[:, 1]) == pytest.approx(SHIFT_Y, abs=0.75)


def test_inliers_are_the_matches_consistent_with_one_affine_model(matched):
    kp_ref, kp_mov, matches, _, inlier_mask, reproj_errors = matched
    inlier_mask = np.asarray(inlier_mask)
    reproj_errors = np.asarray(reproj_errors)
    assert inlier_mask.sum() >= 4
    assert np.all(reproj_errors[inlier_mask] <= config.RANSAC_REPROJ_PATCH * 2.0)

    d = np.array([np.subtract(kp_ref[m.queryIdx].pt, kp_mov[m.trainIdx].pt)
                  for m in matches])
    spread_in = np.std(d[inlier_mask], axis=0).max()
    assert spread_in < 1.0, "inliers should agree on the single true translation"


def test_match_backward_swaps_the_two_sides(matcher, shifted_patches):
    """``match_backward``'s first list is the MOV side, per ``refine.py``."""
    img_ref, img_mov = shifted_patches
    kp_mov, kp_ref, matches, _, inlier_mask, _ = matcher.match_backward(
        img_ref, img_mov)
    idx = np.where(np.asarray(inlier_mask))[0]
    assert len(idx) >= 4

    d = np.array([np.subtract(kp_mov[matches[i].queryIdx].pt,
                              kp_ref[matches[i].trainIdx].pt) for i in idx])
    assert np.median(d[:, 0]) == pytest.approx(-SHIFT_X, abs=0.75)
    assert np.median(d[:, 1]) == pytest.approx(-SHIFT_Y, abs=0.75)


def test_featureless_input_returns_an_empty_but_well_formed_tuple(matcher):
    flat = np.full((PATCH, PATCH), 128, dtype=np.uint8)
    kp_ref, kp_mov, matches, ratios, inlier_mask, reproj_errors = matcher.match(
        flat, flat.copy())
    assert len(matches) == len(ratios) == len(inlier_mask) == len(reproj_errors)
    assert np.asarray(inlier_mask).sum() == 0


def test_multichannel_input_is_handled_and_keeps_the_contract(matcher):
    pad = 32
    base = _synth.texture(PATCH + 2 * pad, PATCH + 2 * pad, seed=7, bands=3)
    base = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    img_ref = np.ascontiguousarray(base[pad:pad + PATCH, pad:pad + PATCH])
    img_mov = np.ascontiguousarray(base[pad + SHIFT_Y:pad + SHIFT_Y + PATCH,
                                        pad + SHIFT_X:pad + SHIFT_X + PATCH])
    kp_ref, kp_mov, matches, ratios, inlier_mask, reproj_errors = matcher.match(
        img_ref, img_mov)
    assert len(matches) == len(ratios) == len(inlier_mask) == len(reproj_errors)
    idx = np.where(np.asarray(inlier_mask))[0]
    assert len(idx) >= 4
    d = np.array([np.subtract(kp_ref[matches[i].queryIdx].pt,
                              kp_mov[matches[i].trainIdx].pt) for i in idx])
    assert np.median(d[:, 0]) == pytest.approx(SHIFT_X, abs=0.75)
    assert np.median(d[:, 1]) == pytest.approx(SHIFT_Y, abs=0.75)


# --------------------------------------------------------------------------
# substitutability: an out-of-package matcher must drop in
# --------------------------------------------------------------------------

class DelegatingMatcher(FeatureMatcher):
    """An external matcher: it knows only the interface, not the package.

    It happens to delegate to RootSIFT so the numbers can be compared, but the
    refine loop cannot tell — it only ever calls ``match`` / ``match_backward``.
    """

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def match(self, img_ref, img_mov):
        self.calls += 1
        return tuple(self._inner.match(img_ref, img_mov))

    def match_backward(self, img_ref, img_mov):
        self.calls += 1
        return tuple(self._inner.match_backward(img_ref, img_mov))


class NullMatcher(FeatureMatcher):
    """A well-formed matcher that never finds anything."""

    def match(self, img_ref, img_mov):
        return [], [], [], [], np.zeros(0, dtype=bool), np.zeros(0)

    def match_backward(self, img_ref, img_mov):
        return self.match(img_mov, img_ref)


def test_the_abstract_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        FeatureMatcher()


def test_rootsift_is_a_featurematcher():
    assert isinstance(RootSIFTMatcher(10, 0.7, 3.0, 3.0), FeatureMatcher)


@pytest.mark.slow
def test_an_external_matcher_drives_the_refine_loop_identically(small_pair):
    from geocoreg import init, refine

    inner = RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                            config.RANSAC_REPROJ_PATCH,
                            config.MULTICHAN_DEDUP_RADIUS)
    external = DelegatingMatcher(
        RootSIFTMatcher(config.SIFT_NFEATURES, config.RATIO_THRESH,
                        config.RANSAC_REPROJ_PATCH,
                        config.MULTICHAN_DEDUP_RADIUS))

    ctx, state = init.run(small_pair.ref_path, small_pair.mov_path)
    builtin = refine.run(ctx, state, inner, diag_folder=None)
    swapped = refine.run(ctx, state, external, diag_folder=None)

    assert external.calls > 0, "refine never called the substituted matcher"
    assert np.array_equal(builtin.D, swapped.D)
    assert np.array_equal(builtin.F, swapped.F)


@pytest.mark.slow
def test_a_matcher_that_finds_nothing_degrades_instead_of_crashing(small_pair):
    from geocoreg import init, refine

    ctx, state = init.run(small_pair.ref_path, small_pair.mov_path)
    out = refine.run(ctx, state, NullMatcher(), diag_folder=None)
    assert out.D.shape == state.D.shape
    assert np.isfinite(out.D).all()
