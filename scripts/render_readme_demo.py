"""Regenerate README figures from the same 3D scene and preparation pipeline.

Run from the repository root: uv run python scripts/render_readme_demo.py
Figures are synthetic input/preparation demonstrations, not Marble API results.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from marble_inpainting.demo import run_demo
from marble_inpainting.demo_scene import (
    CAMERA_POSITIONS,
    CHAIR,
    demo_camera,
    render_scene,
)
from marble_inpainting.geometry import lift_edit_region, project_visible_points
from marble_inpainting.masks import overlay_region, rasterize_fill_region

INK = (35, 47, 57)
MUTED = (87, 102, 112)
BG = (248, 249, 247)
TEAL = (0, 139, 151)


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    size: int = 24,
    color: tuple[int, int, int] = INK,
) -> None:
    draw.text(xy, label, font=ImageFont.load_default(size=size), fill=color)


def picture(
    canvas: Image.Image, path: Path, xy: tuple[int, int], size: tuple[int, int]
) -> None:
    with Image.open(path) as source:
        canvas.paste(source.convert("RGB").resize(size, Image.Resampling.LANCZOS), xy)


def multiview_figure(root: Path, destination: Path) -> None:
    canvas = Image.new("RGB", (1440, 990), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (24, 18), "ONE EDIT REGION. THREE CAMERA VIEWS.", 32)
    labels = ("LEFT / view_00", "ANCHOR / view_01", "RIGHT / view_02")
    for index, label in enumerate(labels):
        text(draw, (24 + index * 472, 77), label, 25)
    rows = (
        ("Original RGB inputs", "source/rgb", 120),
        (
            "Prepared context: red = fill, cyan = trusted anchor edit",
            "prepared/previews",
            410,
        ),
        ("API keep masks: black = fill, white = keep", "prepared/masks", 700),
    )
    for title, relative, y in rows:
        text(draw, (24, y), title, 23, MUTED)
        paths = sorted((root / relative).glob("*.png"))
        paths = [path for path in paths if path.name != "contact-sheet.png"]
        for index, path in enumerate(paths):
            picture(canvas, path, (24 + index * 472, y + 35), (448, 252))
    canvas.save(destination, optimize=True)


def geometry_figure(root: Path, points: np.ndarray, destination: Path) -> None:
    canvas = Image.new("RGB", (1440, 540), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (24, 18), "RGB PIXELS + DEPTH + CAMERA  ->  3D SURFACE POINTS", 30)
    text(draw, (24, 74), "Anchor camera z-depth", 24)
    text(draw, (720, 74), "Lifted chair surface, seen from another angle", 24)
    depth = np.load(root / "source/depth/view_01.npy", allow_pickle=False)
    t = np.clip((depth - 3.5) / 6.0, 0, 1)
    near, far = np.array([255, 208, 91]), np.array([42, 58, 111])
    rgb = ((1 - t[..., None]) * near + t[..., None] * far).astype(np.uint8)
    canvas.paste(Image.fromarray(rgb).resize((640, 360)), (24, 112))
    text(draw, (24, 484), "Near: gold    /    Far: blue", 23, MUTED)

    # Orthographic diagram of actual lifted points, with a ground grid.
    eye = demo_camera((3.5, -2.8, -4.0)).camera_to_world_rdf()
    center = np.array([0, -0.9, 0.3])

    def screen(xyz: np.ndarray) -> np.ndarray:
        local = (xyz - center) @ eye
        return local[:, :2] * 145 + np.array([1050, 280])

    for x in np.linspace(-1.25, 1.25, 7):
        ends = screen(np.array([[x, 0, -0.75], [x, 0, 0.75]]))
        draw.line([tuple(p) for p in ends], fill=(214, 222, 217), width=1)
    for z in np.linspace(-0.75, 0.75, 5):
        ends = screen(np.array([[-1.25, 0, z], [1.25, 0, z]]))
        draw.line([tuple(p) for p in ends], fill=(214, 222, 217), width=1)
    xy = screen(points)
    for x, y in xy:
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=TEAL)
    text(
        draw,
        (720, 484),
        "Only surfaces visible in the anchor are available.",
        22,
        MUTED,
    )
    canvas.save(destination, optimize=True)


def orbit_animation(points: np.ndarray, destination: Path) -> None:
    frames = []
    for camera_x in np.linspace(-3.2, 3.2, 25):
        camera = demo_camera((float(camera_x), -2.4, -5.7), width=480, height=270)
        rendered = render_scene(camera)
        projection = project_visible_points(
            points, camera, rendered.depth, depth_relative_tolerance=0.02
        )
        region = rasterize_fill_region(
            projection.visible_xy,
            height=270,
            width=480,
            footprint_radius_px=1,
            closing_radius_px=0,
            margin_px=1,
        )
        overlay = overlay_region(
            rendered.rgb, region, color=(255, 64, 64), label="Projected edit"
        )
        canvas = Image.new("RGB", (1008, 380), BG)
        draw = ImageDraw.Draw(canvas)
        text(draw, (16, 14), "Move the camera around the same 3D scene", 25)
        text(draw, (16, 57), "Original RGB", 21, MUTED)
        text(draw, (512, 57), "Projected anchor region (red)", 21, MUTED)
        canvas.paste(Image.fromarray(rendered.rgb), (16, 88))
        canvas.paste(Image.fromarray(overlay), (512, 88))
        frames.append(canvas)
        print(f"Rendered orbit frame {len(frames)}/25", flush=True)
    frames += list(reversed(frames[1:-1]))
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="marble-readme-") as temporary:
        root = run_demo(Path(temporary) / "demo")
        files = {
            "source/rgb/view_01.png": "input-original-anchor.png",
            "source/edited-anchor.png": "input-edited-anchor.png",
            "source/edit-region.png": "input-edit-region.png",
            "prepared/previews/contact-sheet.png": "output-projected-regions.png",
            "prepared/masks/000-view_00.keep.png": "output-keep-view-00.png",
            "prepared/masks/001-view_01.keep.png": "output-keep-anchor.png",
            "prepared/masks/002-view_02.keep.png": "output-keep-view-02.png",
        }
        for source, filename in files.items():
            shutil.copyfile(root / source, args.output / filename)
        anchor = demo_camera(CAMERA_POSITIONS[1])
        rendered = render_scene(anchor)
        lifted = lift_edit_region(
            rendered.object_ids == CHAIR,
            rendered.depth,
            anchor,
            max_points=16000,
            min_valid_pixels=64,
            seed=0,
        )
        multiview_figure(root, args.output / "multiview-output.png")
        geometry_figure(root, lifted.points_world, args.output / "depth-to-3d.png")
        orbit_animation(lifted.points_world, args.output / "camera-orbit.gif")
        print(f"Wrote README assets to {args.output}")


if __name__ == "__main__":
    main()
