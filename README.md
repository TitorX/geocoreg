# geocoreg

Dense, per-pixel co-registration of georeferenced satellite imagery — the
reference implementation of SCDF (self-calibrating dense displacement fields),
described in [arXiv:2608.22300](https://arxiv.org/abs/2608.22300).

Given a reference raster (`REF`) and a moving raster (`MOV`), both already
georeferenced, `geocoreg` estimates for **every pixel of `MOV`** where that
pixel belongs in geographic space, and resamples `MOV` accordingly. The result
is a dense displacement field rather than one global shift or homography, so
deformation that varies across a scene (a seam, a local warp, a terrain-driven
residual) is resolved instead of averaged away.

Every data-dependent acceptance threshold in the pipeline is calibrated from
the pair being registered — its own residual distribution, its own correlation null,
its own cross-validated interpolator. There is nothing to tune per scene, and
the defaults in `config.py` are what all published results were produced with.

## Install

GDAL's Python bindings (`osgeo`) are a hard requirement and are deliberately
not declared in `pyproject.toml`: installing GDAL through pip needs a matching
system GDAL and usually fails. Install it first, from conda-forge or pixi:

```bash
conda install -c conda-forge gdal      # or: pixi add gdal
python -c "from osgeo import gdal; print(gdal.__version__)"
```

Then the package itself:

```bash
git clone https://github.com/TitorX/geocoreg.git
cd geocoreg
pip install -e .            # optional extras: ".[viz]" (matplotlib figures), ".[dev]" (pytest)
```

Python ≥ 3.10; `numpy`, `opencv-python-headless`, `scipy` and `scikit-image`
are installed automatically. Without `matplotlib` the pipeline runs normally
and simply skips the diagnostic figures. The published results were produced
with Python 3.13, GDAL 3.12, OpenCV 4.13, numpy 2.4, SciPy 1.18,
scikit-image 0.26.

## Use

```bash
geocoreg match --ref ref.tif --mov mov.tif --out out/
```

That is the whole invocation — no per-pair parameters. `out/` then holds:

```
matched.tif                MOV resampled onto its corrected geolocation
control_points.geojson     every measured correspondence, with confidence
control_points.points      the same points as a GCP text file (QGIS Georeferencer format)
log.txt                    run log (levels, gates, point counts)
*.png, patches/            diagnostic figures (only with matplotlib installed)
```

To replay an existing set of control points onto another raster:

```bash
geocoreg apply --ref ref.tif --mov mov.tif --out out/ --cp control_points.geojson
```

The same two entry points are available from Python:

```python
from geocoreg import run_match, run_apply

run_match("ref.tif", "mov.tif", "out/")
```

`matched.tif` follows the `REF`/`MOV` overlap on `MOV`'s grid, keeps `MOV`'s
radiometry (band count, dtype, nodata), and resamples with nearest neighbor by
default so original digital numbers survive.

Before consuming the displacement field programmatically, read
[docs/concepts.md](docs/concepts.md) — the sign, the grid and the units are all
conventions that fail silently when misread. [docs/how-it-works.md](docs/how-it-works.md)
summarizes the pipeline itself. Settings live in `src/geocoreg/config.py`, as
module-level constants that can also be overridden at runtime:

```python
import geocoreg.config as cfg
cfg.WARP_INTERP_METHOD = "cubic"
```

## Layout

```
src/geocoreg/       the library: init -> refine -> warp, plus the matcher plug point
tests/              pytest suite; inputs are generated procedurally, no imagery needed
docs/               the displacement convention and a pipeline overview
```

Run the tests with `pytest tests` (add `-m "not slow"` to skip the end-to-end
cases).

This repository is the method only. The benchmark imagery used in the paper,
with the script that regenerates its synthetic constructions, is archived on
Zenodo under CC BY 4.0:
[doi:10.5281/zenodo.21900160](https://doi.org/10.5281/zenodo.21900160). The
evaluation and experiment-orchestration code is not part of the release.

## License

MIT, see `LICENSE`.
