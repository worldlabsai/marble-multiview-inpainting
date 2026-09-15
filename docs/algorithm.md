# Projection algorithm

This document specifies the automatic preparation path. The implementation is
split into importable functions so callers can inspect or replace each stage.

## 1. Normalize pixels and cameras

Each RGB, depth, and mask grid is scale-to-cover resized and center-cropped to
the requested output size. The identical pixel transform is applied to all
modalities and the intrinsics are updated as described in
[input-format.md](input-format.md).

## 2. Lift the anchor edit region

For edit pixel `(x, y)` with valid camera-space z-depth `z`, sample the pixel
center:

```text
u = x + 0.5
v = y + 0.5
Xrdf = [(u - cx) * z / fx, (v - cy) * z / fy, z]
```

The camera-to-world rotation and position transform `Xrdf` to world space. To
match Marble's camera conversion exactly, a RUB homogeneous pose `C` is
canonicalized as `F @ C @ F`, where `F = diag(1, -1, -1, 1)`. RDF poses pass
through unchanged.

If more than `max_points` edit pixels have valid depth, a deterministic seeded
sample is used. Preparation fails when the anchor region is empty or has too
few valid-depth pixels.

## 3. Project and check visibility

For each target camera, world points are transformed to its internal RDF camera
frame. Points behind the camera or outside the image are rejected. Remaining
points project as:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

The projected point is visible only when target-view depth is valid and:

```text
abs(target_depth - Z) <= depth_relative_tolerance * target_depth
```

This removes points hidden by target-view foreground surfaces. A region that is
entirely out of frame or fully occluded receives an all-white keep mask and a
report warning. Partially observed projections with insufficient valid depth or
too few visible samples fail and request a manual override.

## 4. Rasterize and expand

Visible integer pixels are rasterized, dilated by `footprint_radius_px`, closed
by `closing_radius_px`, hole-filled, and then dilated by `margin_px`. These
operations turn a sampled point footprint into a conservative fill region.

Every morphology parameter is expressed in output pixels and recorded in
`report.json`.

## 5. Convert to API keep masks

The internal Boolean region uses `true = fill`. It is converted exactly once to
the API convention:

```text
inside fill region: keep = 0
outside fill region: keep = 255
```

When `feather_px` is positive, pixels immediately outside the fill region ramp
from 0 to 255 according to Euclidean distance. The core fill region remains 0.

The edited anchor is a special case: its keep mask is all 255 because the full
edited image is observed context. Its original edit region is shown in cyan in
the preview only.

## Defaults

Run the code rather than copying defaults from prose:

```bash
./marble-inpaint config
```

The command prints every effective default. The same values are embedded in
each output report, preventing README drift from changing the interpretation of
an existing prepared dataset.
