# How it works

Three phases — initialize, refine, warp. The first two produce the dense
displacement field described in [concepts.md](concepts.md); the third turns it
into a raster and is optional.

```
REF, MOV ──▶ init.py     overlap, GSD ratio, empty D/F/C state, one seed point
             refine.py   coarse-to-fine patch matching, then the fill
             warp.py     GDAL geolocation arrays → matched.tif
```

Three arrays are carried through refinement, all on the `MOV` overlap grid:
`D`, the displacement field; `F`, whether a pixel carries a measured match
rather than an interpolated value; and `C`, a confidence on one scale shared by
every stage that produces points.

## Init (`init.py`)

Both rasters are opened, the CRS transforms built in both directions, and their
geographic footprints intersected (no overlap is a hard error). The overlap
becomes a pixel crop window on each raster, and the **GSD ratio** — how many
`MOV` pixels one `REF` pixel spans — is estimated at its center to drive patch
sizing; it is re-estimated per patch later, since it is not constant across a
scene when the CRSs differ. `D`/`F`/`C` are allocated and seeded with a single
zero-displacement point at the overlap center, at low confidence: the
assumption that both rasters are roughly where their metadata says.

## Refine (`refine.py`)

Patch size starts at the whole overlap and halves each level until both sides
fall below `MIN_PATCH_SIZE`, with patch centers tiling the overlap at every
level. Coarse levels exist to survive a large initial misalignment — a 256 px
patch cannot find its counterpart across a 500 px offset, but a patch covering
the whole overlap can — and each level hands the next a better prior.

Per level: a moving-least-squares prior is interpolated at each patch center
from the control points accepted so far; patches that already contain
`SPARSE_THRESHOLD` measured points are skipped so later levels work on the
still-empty areas; the prior and the local GSD ratio locate and size the `REF`
search window; the finer side is downsampled to the common resolution and both
are capped at `MAX_DOWNSAMPLE_SIDE`; then RootSIFT matching with a Lowe ratio
test and a local RANSAC affine gate. The patch is matched again in reverse and
each inlier must return to within `FWD_BWD_CONSISTENCY_PX` of where it started,
which is what rejects the asymmetric matches that repetitive texture produces.
Each survivor becomes a control point with confidence = ratio-test margin ×
reprojection-error term.

New points are then filtered by a global magnitude MAD test, a leave-one-out
local-affine residual filter (`filters.py` — the model is affine, so a smooth
displacement gradient fits exactly and valid points on a ramp survive), and
non-maximum suppression at `NMS_RADIUS` to bound density.

**Subpixel measurement** (`subpixel.py`) runs after the pyramid. Feature
keypoints localize to roughly 0.3–0.5 px of the grid they were matched on,
which on a cross-resolution pair is a dozen fine pixels, so every point is
re-measured by upsampled cross-correlation against the `REF` window the current
field predicts. A second stage, densification, runs the same measurement on a
uniform lattice seeded from the interpolated prior, adding points where the
feature matcher found nothing. A point that fails the acceptance gates keeps
its feature-matched displacement but is downweighted, not deleted — deleting
them would remove the anchors at displacement discontinuities, where the field
is least predictable from its neighbors. The MAD and local-affine filters then
run once more.

Remaining pixels are interpolated from the accepted control points, with the
interpolator chosen per pair by 5-fold cross-validation over that pair's own
points (MLS at several `k`/`α`, IDW, thin-plate spline) unless
`INTERP_AUTOTUNE` is off.

## Self-calibration

Four decisions that a registration method would normally hard-code are instead
calibrated on the pair being registered:

- **outlier rejection** — the local-affine threshold is a robust scale of this
  pair's residual distribution;
- **correlation acceptance** — a window is accepted only if its response beats
  the same `REF` window correlated against deliberately mismatched partners, so
  a smooth window is judged against its own high junk baseline and a textured
  one against its low baseline;
- **error gating** — a quantile of this pair's error distribution, since
  correlation NRMS measures radiometric similarity rather than lock quality;
- **interpolator selection** — cross-validated, as above.

Structural constants remain (Lowe's ratio, the RANSAC reprojection threshold,
the pyramid floor); they are properties of the algorithms, not of the data.

## Warp (`warp.py`)

The field is applied through GDAL's geolocation-array mechanism rather than a
per-pixel resampling loop. Two Float64 rasters over the `MOV` overlap grid hold
the corrected position of each source pixel's center
(`X_geo = pix2geo(col + 0.5, row + 0.5).x + D[..., 0]`, likewise for Y, with
non-finite entries written as nodata). They are attached to a VRT view of the
`MOV` overlap as `GEOLOCATION` metadata with `PIXEL_OFFSET = LINE_OFFSET = 0.5`
— GDAL maps a pixel coordinate `P` to index `(P − PIXEL_OFFSET) / PIXEL_STEP`,
so center-valued arrays need the half-pixel offset or the whole output shifts
— and `gdal.Warp(..., geoloc=True)` does the resampling. CRS is not handled
here at all: the conversion was baked into `D` during refinement.

## The matcher is a plug point

Nothing in the refine loop, the filters or the correlation stages depends on
how correspondences were produced. They arrive through
`geocoreg.matchers.FeatureMatcher`, whose `match(img_ref, img_mov)` and
`match_backward(...)` both return `(kp_ref, kp_mov, good_matches, ratios,
inlier_mask, reproj_errors)` for `uint8` input of shape `(H, W)` or `(H, W, C)`
— see `matchers/base.py`. `RootSIFTMatcher` is the implementation shipped here
and the one the pipeline builds; another matcher only needs to implement this
interface.
