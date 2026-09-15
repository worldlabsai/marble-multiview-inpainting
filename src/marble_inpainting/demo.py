"""Generate a small synthetic RGBD scene and run the full preparation path."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from marble_inpainting.demo_scene import (
    CAMERA_POSITIONS,
    CHAIR,
    demo_camera,
    render_scene,
)
from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.models import PrepareConfig
from marble_inpainting.pipeline import prepare_auto
from marble_inpainting.scene_io import write_json

DEMO_MARKER = ".marble-inpaint-demo-output"


def _write_demo_inputs(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source"
    rgb_dir = source / "rgb"
    depth_dir = source / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    views = []
    anchor_edited: np.ndarray | None = None
    anchor_mask: np.ndarray | None = None
    for index, position in enumerate(CAMERA_POSITIONS):
        view_id = f"view_{index:02d}"
        camera = demo_camera(position)
        rendered = render_scene(camera)
        image_relative = Path("rgb") / f"{view_id}.png"
        depth_relative = Path("depth") / f"{view_id}.npy"
        Image.fromarray(rendered.rgb, mode="RGB").save(source / image_relative)
        np.save(source / depth_relative, rendered.depth)
        if index == 1:
            anchor_edited = rendered.edited_rgb
            anchor_mask = rendered.object_ids == CHAIR
        views.append(
            {
                "id": view_id,
                "image": str(image_relative),
                "depth": {
                    "path": str(depth_relative),
                    "representation": "camera_z",
                },
                "camera": camera.to_dict(),
            }
        )

    if anchor_edited is None or anchor_mask is None:  # pragma: no cover
        raise RuntimeError("demo anchor was not generated")
    edited_path = source / "edited-anchor.png"
    mask_path = source / "edit-region.png"
    Image.fromarray(anchor_edited, mode="RGB").save(edited_path)
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
            prompt="Change the lounge chair upholstery from terracotta to teal",
            output=staging / "prepared",
            # Exact synthetic depth permits a tighter mask than noisy real scenes.
            config=PrepareConfig(
                max_points=24000,
                depth_relative_tolerance=0.02,
                footprint_radius_px=2,
                closing_radius_px=1,
                margin_px=4,
                feather_px=2,
            ),
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
