# The displacement field

Everything the library produces is defined by one equation on one grid. Read
this before consuming the output programmatically — misread conventions do not
raise exceptions.

`REF` is the reference and is never modified; `MOV` is the image being
corrected. The question answered is always *where should each `MOV` pixel move
to*, so `REF`'s resolution, grid and CRS never affect the shape of the result.

For the `MOV` pixel at row `y`, column `x`:

```
D_geo[y, x, :] = corrected_mov_geo - original_mov_geo
```

- **Shape** `(mov_height, mov_width, 2)` — `MOV`'s own pixel grid, indexed
  `(y, x)` as numpy indexes an array (GDAL's raster convention).
- **Channel 0 is the x displacement, channel 1 the y displacement**, matching
  the `(x, y)` order of a geotransform rather than the `(y, x)` order of the
  index.
- **Units are `MOV`'s CRS units** — meters for UTM, degrees for a geographic
  CRS. Divide by the geotransform's pixel size for pixel units (`gt[5]` is
  negative for a north-up raster).
- **The sign is `corrected − original`**, so adding the field to the original
  position gives the corrected one; zeros mean "already aligned".
- **CRS conversion is already baked in.** If `REF` and `MOV` differ, the
  matched `REF` position is converted to `MOV`'s CRS before the difference is
  taken, so the field is wholly in `MOV`'s CRS and needs no further
  reprojection.

## Pixel centers

A GDAL geotransform maps an integer pixel index to a pixel's **top-left
corner**, so the center of pixel `(row, col)` is at `(col + 0.5, row + 0.5)`:

```python
x_geo = gt[0] + (col + 0.5) * gt[1] + (row + 0.5) * gt[2]
y_geo = gt[3] + (col + 0.5) * gt[4] + (row + 0.5) * gt[5]
```

Every pixel-to-geographic conversion here uses centers. At equal resolution a
consistent corner convention cancels in the `REF − MOV` difference; across
resolutions it does not, leaving a bias of order `0.5 × |GSD_ref − GSD_mov|` —
several fine pixels on a 10 m / 0.6 m pair, which no matching quality removes.

## Validity

A value exists at every pixel but is not meaningful everywhere. Four things
make one invalid: the pixel is outside the `REF`/`MOV` overlap; `MOV` has no
usable content there (nodata, cloud, shadow); matching found nothing to lock
onto and the gates declined rather than accepting a wrong measurement; or the
value comes from extrapolating far beyond any control point.

The last two do not look like failures — the value is present, smooth and
wrong. The field is trustworthy where control points are dense and degrades
with distance from them, and `control_points.geojson` is what shows where they
are. In `matched.tif` invalid regions carry `MOV`'s nodata value, which a
raster that declares no nodata cannot express — tag nodata on the inputs.

## Comparing two fields

Two fields for the same `MOV` are automatically pixel-aligned under this
convention, so a predicted field and a reference field can be subtracted
directly. Assert grid equality (width, height, geotransform, CRS) instead of
resampling one onto the other: resampling introduces an error of the same order
as the quantity being measured, silently.
