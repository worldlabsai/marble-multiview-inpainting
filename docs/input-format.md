# Input format

## Scene manifest

The scene manifest is UTF-8 JSON with `schemaVersion: 1` and an ordered `views`
array. View order is preserved in every output and in the `atlasMasked` request
template.

Each view requires:

- a unique string `id`;
- a local RGB `image` path;
- a Marble pinhole `camera`; and
- in automatic mode, a local `depth` path with `representation: "camera_z"`.

Paths are resolved relative to the manifest. `.npy`, `.npz`, and linear-depth
`.exr` are supported. An NPZ may contain one array or an array named `depth`.
For a multichannel EXR, the first decoded channel is used, matching Marble's
linear-depth loader.

## Pixel grid

The RGB image dimensions must equal `camera.intrinsics.width` and `height`.
Depth and edit-region masks must have that same height and width. Inputs may use
an arbitrary resolution; outputs default to 1280 x 720.

Normalization uses scale-to-cover followed by an integer center crop. RGB uses
Lanczos resampling. Masks and depth use nearest-neighbor resampling to avoid
inventing fractional labels or interpolating across depth discontinuities.

## Intrinsics

`fx`, `fy`, `cx`, and `cy` are pixel values. Pixel `(0, 0)` is at the image's
top left, +x points right, and +y points down.

If a source grid is resized to `(Wr, Hr)` and center-cropped at `(x0, y0)`, the
tool applies:

```text
sx = Wr / W                 sy = Hr / H
fx' = sx * fx               fy' = sy * fy
cx' = sx * cx - x0          cy' = sy * cy - y0
```

## Extrinsics

The pose is camera-to-world. Quaternions use XYZW order and are normalized on
read. `coordinateSystem` is:

| Value | Camera axes | Forward direction |
| --- | --- | --- |
| `rdf` | right, down, forward | +Z |
| `rub` | right, up, back | -Z |

The tool converts both conventions to Marble's canonical RDF frame before
unprojection or projection. A RUB camera-to-world matrix `C` is converted as
`F @ C @ F`, where `F = diag(1, -1, -1, 1)`. This transforms both orientation
and position. Changing only the string does not convert a pose; the supplied
quaternion and position must already be expressed in the declared convention.

All cameras must share one world frame and scale.

## Depth

Depth is positive camera-space z-depth along the camera's forward axis, not
Euclidean ray length. It uses the same scale as camera positions. Non-finite,
zero, and negative values are invalid.

OpenCV decodes EXR into top-to-bottom NumPy rows, so this tool does not apply the
row flip needed by Three.js `EXRLoader` CPU arrays. Do not pre-flip an EXR for
this tool.

## Edit-region masks

Edit-region masks may be a grayscale image or RGB/RGBA with identical RGB
channels. Values of 128 or greater belong to the edit region. Differing RGB
channels are rejected as ambiguous.

The anchor mask describes the changed region for geometric projection and
previewing. It is not sent to Marble. The anchor's generated API keep mask is
all white because the complete edited anchor image is trusted context.
