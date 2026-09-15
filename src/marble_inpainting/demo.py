"""Generate a small synthetic RGBD scene and run the full preparation path."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.models import PrepareConfig
from marble_inpainting.pipeline import prepare_auto
from marble_inpainting.scene_io import write_json

DEMO_MARKER = ".marble-inpaint-demo-output"


def _background(x_world: np.ndarray, y_world: np.ndarray) -> np.ndarray:
    checker = ((np.floor(x_world * 2) + np.floor(y_world * 2)) % 2) != 0
    horizontal = np.clip((x_world + 4.0) / 8.0, 0.0, 1.0)
    vertical = np.clip((y_world + 2.5) / 5.0, 0.0, 1.0)
    rgb = np.empty((*x_world.shape, 3), dtype=np.float32)
    rgb[..., 0] = 55 + 35 * horizontal + checker * 16
    rgb[..., 1] = 90 + 55 * vertical + checker * 12
    rgb[..., 2] = 135 + 35 * (1 - horizontal) + checker * 10
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_view(
    *,
    camera_x: float,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    ray_x = (xx + 0.5 - cx) / fx
    ray_y = (yy + 0.5 - cy) / fy

    background_x = camera_x + ray_x * 5.0
    background_y = ray_y * 5.0
    rgb = _background(background_x, background_y)
    background_only = rgb.copy()

    object_x = camera_x + ray_x * 3.0
    object_y = ray_y * 3.0
    object_mask = (np.abs(object_x) <= 0.58) & (np.abs(object_y) <= 0.43)
    depth = np.full((height, width), 5.0, dtype=np.float32)
    depth[object_mask] = 3.0

    object_color = np.empty_like(rgb)
    object_color[..., 0] = 198
    object_color[..., 1] = 74 + np.clip((object_y + 0.43) * 45, 0, 38).astype(np.uint8)
    object_color[..., 2] = 54
    rgb[object_mask] = object_color[object_mask]
    border = object_mask & ~((np.abs(object_x) <= 0.53) & (np.abs(object_y) <= 0.38))
    rgb[border] = np.array([245, 184, 72], dtype=np.uint8)
    return rgb, depth, background_only


def _write_demo_inputs(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source"
    rgb_dir = source / "rgb"
    depth_dir = source / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    width, height = 640, 360
    fx = fy = 450.0
    cx, cy = width / 2, height / 2
    positions = (-0.35, 0.0, 0.35)
    views = []
    anchor_background: np.ndarray | None = None
    anchor_mask: np.ndarray | None = None
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2  # pylint: disable=import-outside-toplevel

    for index, camera_x in enumerate(positions):
        view_id = f"view_{index:02d}"
        rgb, depth, background = _render_view(
            camera_x=camera_x,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        image_relative = Path("rgb") / f"{view_id}.png"
        depth_relative = Path("depth") / f"{view_id}.exr"
        Image.fromarray(rgb, mode="RGB").save(source / image_relative)
        if not cv2.imwrite(str(source / depth_relative), depth):
            raise RuntimeError(f"could not write demo EXR: {depth_relative}")
        if index == 1:
            anchor_background = background
            anchor_mask = depth == 3.0
        views.append(
            {
                "id": view_id,
                "image": str(image_relative),
                "depth": {
                    "path": str(depth_relative),
                    "representation": "camera_z",
                },
                "camera": {
                    "extrinsics": {
                        "position": [camera_x, 0.0, 0.0],
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                        "coordinateSystem": "rdf",
                    },
                    "intrinsics": {
                        "width": width,
                        "height": height,
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                    },
                },
            }
        )

    if anchor_background is None or anchor_mask is None:  # pragma: no cover
        raise RuntimeError("demo anchor was not generated")
    edited_path = source / "edited-anchor.png"
    mask_path = source / "edit-region.png"
    Image.fromarray(anchor_background, mode="RGB").save(edited_path)
    Image.fromarray(anchor_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    manifest_path = source / "scene.json"
    write_json(manifest_path, {"schemaVersion": 1, "views": views})
    return manifest_path, edited_path, mask_path


def run_demo(output: str | Path, *, overwrite: bool = False) -> Path:
    """Create demo inputs and prepared outputs under one safe directory."""

    destination = Path(output).expanduser().resolve()
    if destination == destination.parent:
        raise MarbleInpaintError("demo output cannot be a filesystem root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise MarbleInpaintError(f"demo output path is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise MarbleInpaintError(
                f"demo output is not empty: {destination}; pass --overwrite"
            )
        if not (destination / DEMO_MARKER).is_file():
            raise MarbleInpaintError(
                f"refusing to overwrite {destination}; marker is missing "
                f"({DEMO_MARKER})"
            )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        manifest, edited, mask = _write_demo_inputs(staging)
        prepare_auto(
            scene=manifest,
            anchor_view="view_01",
            edited_anchor=edited,
            edit_region=mask,
            prompt="Remove the red panel and continue the patterned wall",
            output=staging / "prepared",
            config=PrepareConfig(),
        )
        (staging / DEMO_MARKER).write_text("marble-inpaint-demo-v1\n", encoding="utf-8")
        if destination.exists():
            if any(destination.iterdir()):
                shutil.rmtree(destination)
            else:
                destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
