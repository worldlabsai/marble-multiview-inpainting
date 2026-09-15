"""Read local scene manifests, pixels, masks, and linear-depth arrays."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.models import Camera, Scene, View


def _resolve_file(base: Path, raw_path: Any, *, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise MarbleInpaintError(f"{field} must be a non-empty local path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise MarbleInpaintError(f"{field} does not exist or is not a file: {path}")
    return path


def load_scene(path: str | Path) -> Scene:
    """Load schema version 1 of the local multiview scene manifest."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise MarbleInpaintError(f"scene manifest does not exist: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarbleInpaintError(f"could not read scene manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise MarbleInpaintError("scene manifest must contain a JSON object")
    unknown_top_level = sorted(set(data) - {"$schema", "schemaVersion", "views"})
    if unknown_top_level:
        raise MarbleInpaintError(
            "unknown scene manifest field(s): " + ", ".join(unknown_top_level)
        )
    version = data.get("schemaVersion", 1)
    if version != 1:
        raise MarbleInpaintError(
            f"unsupported scene schemaVersion {version!r}; expected 1"
        )
    raw_views = data.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise MarbleInpaintError("scene.views must be a non-empty array")

    base = manifest_path.parent
    seen_ids: set[str] = set()
    views: list[View] = []
    for index, raw_view in enumerate(raw_views):
        field = f"views[{index}]"
        if not isinstance(raw_view, dict):
            raise MarbleInpaintError(f"{field} must be an object")
        unknown_view_fields = sorted(set(raw_view) - {"id", "image", "depth", "camera"})
        if unknown_view_fields:
            raise MarbleInpaintError(
                f"unknown {field} field(s): " + ", ".join(unknown_view_fields)
            )
        view_id = raw_view.get("id")
        if not isinstance(view_id, str) or not view_id.strip():
            raise MarbleInpaintError(f"{field}.id must be a non-empty string")
        if view_id in seen_ids:
            raise MarbleInpaintError(f"duplicate view id: {view_id!r}")
        seen_ids.add(view_id)

        depth_path: Path | None = None
        depth_representation: str | None = None
        raw_depth = raw_view.get("depth")
        if raw_depth is not None:
            if isinstance(raw_depth, str):
                depth_path = _resolve_file(base, raw_depth, field=f"{field}.depth")
                depth_representation = "camera_z"
            elif isinstance(raw_depth, dict):
                unknown_depth_fields = sorted(
                    set(raw_depth) - {"path", "representation"}
                )
                if unknown_depth_fields:
                    raise MarbleInpaintError(
                        f"unknown {field}.depth field(s): "
                        + ", ".join(unknown_depth_fields)
                    )
                depth_path = _resolve_file(
                    base, raw_depth.get("path"), field=f"{field}.depth.path"
                )
                depth_representation = raw_depth.get("representation", "camera_z")
                if depth_representation != "camera_z":
                    raise MarbleInpaintError(
                        f"{field}.depth.representation must be 'camera_z'"
                    )
            else:
                raise MarbleInpaintError(f"{field}.depth must be a path or an object")

        views.append(
            View(
                id=view_id,
                image_path=_resolve_file(
                    base, raw_view.get("image"), field=f"{field}.image"
                ),
                depth_path=depth_path,
                depth_representation=depth_representation,
                camera=Camera.from_dict(
                    raw_view.get("camera"), field=f"{field}.camera"
                ),
            )
        )
    return Scene(manifest_path=manifest_path, views=tuple(views))


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image without applying EXIF orientation transforms."""

    image_path = Path(path)
    try:
        with Image.open(image_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise MarbleInpaintError(
            f"could not read RGB image {image_path}: {exc}"
        ) from exc


def load_edit_mask(path: str | Path) -> np.ndarray:
    """Load a grayscale mask, accepting RGB only when all channels agree."""

    mask_path = Path(path)
    try:
        with Image.open(mask_path) as image:
            array = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise MarbleInpaintError(f"could not read mask {mask_path}: {exc}") from exc

    if array.ndim == 2:
        pass
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3]
        if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(
            rgb[..., 0], rgb[..., 2]
        ):
            raise MarbleInpaintError(
                f"mask must be grayscale; RGB channels differ in {mask_path}"
            )
        array = rgb[..., 0]
    else:
        raise MarbleInpaintError(
            f"mask must have shape HxW, HxWx3, or HxWx4; got {array.shape}"
        )

    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(array)
        if not finite.all():
            raise MarbleInpaintError(f"mask contains non-finite values: {mask_path}")
        maximum = float(array.max(initial=0))
        if maximum <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def load_depth(path: str | Path) -> np.ndarray:
    """Load camera-space z-depth from NPY, NPZ, or linear-depth EXR."""

    depth_path = Path(path)
    suffix = depth_path.suffix.lower()
    try:
        if suffix == ".npy":
            depth = np.load(depth_path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(depth_path, allow_pickle=False) as archive:
                if "depth" in archive.files:
                    depth = archive["depth"]
                elif len(archive.files) == 1:
                    depth = archive[archive.files[0]]
                else:
                    raise MarbleInpaintError(
                        f"NPZ must contain one array or a 'depth' array: {depth_path}"
                    )
        elif suffix == ".exr":
            import OpenEXR  # pylint: disable=import-outside-toplevel

            with OpenEXR.File(str(depth_path), separate_channels=True) as exr_file:
                if len(exr_file.parts) != 1:
                    raise MarbleInpaintError(
                        f"depth EXR must contain exactly one part: {depth_path}"
                    )
                channels = exr_file.channels()
                if len(channels) == 1:
                    depth = next(iter(channels.values())).pixels
                else:
                    channel_name = next(
                        (name for name in ("R", "Y", "Z") if name in channels), None
                    )
                    if channel_name is None:
                        available = ", ".join(sorted(channels))
                        raise MarbleInpaintError(
                            "multichannel depth EXR must contain R, Y, or Z; "
                            f"found {available}: {depth_path}"
                        )
                    depth = channels[channel_name].pixels
        else:
            raise MarbleInpaintError(
                f"unsupported depth format {suffix!r}; use .npy, .npz, or .exr"
            )
    except MarbleInpaintError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise MarbleInpaintError(f"could not read depth {depth_path}: {exc}") from exc

    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[2] < 1:
            raise MarbleInpaintError(f"depth has no channels: {depth_path}")
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise MarbleInpaintError(
            f"depth must have shape HxW or HxWxC; got {depth.shape}: {depth_path}"
        )
    if not np.issubdtype(depth.dtype, np.number):
        raise MarbleInpaintError(f"depth must be numeric: {depth_path}")
    return depth.astype(np.float32, copy=False)


def save_rgb_png(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        path, format="PNG", optimize=True
    )


def save_grayscale_png(path: str | Path, array: np.ndarray) -> None:
    """Write a true 8-bit, single-channel grayscale PNG."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(
        path, format="PNG", optimize=True
    )


def png_header(path: str | Path) -> dict[str, int]:
    """Return dimensions and IHDR encoding fields from a PNG."""

    png_path = Path(path)
    data = png_path.read_bytes()[:29]
    if len(data) < 29 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise MarbleInpaintError(f"not a valid PNG: {png_path}")
    width, height = struct.unpack(">II", data[16:24])
    return {
        "width": width,
        "height": height,
        "bitDepth": data[24],
        "colorType": data[25],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
