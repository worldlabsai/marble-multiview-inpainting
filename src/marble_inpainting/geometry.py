"""Lift an edit region to 3D and reproject it with depth-based visibility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.models import Camera


@dataclass(frozen=True)
class LiftResult:
    points_world: np.ndarray
    edit_pixels: int
    valid_depth_pixels: int
    sampled_points: int


@dataclass(frozen=True)
class ProjectionResult:
    visible_xy: np.ndarray
    total_points: int
    in_bounds_points: int
    valid_target_depth_points: int
    visible_points: int

    @property
    def in_bounds_fraction(self) -> float:
        return self.in_bounds_points / max(self.total_points, 1)

    @property
    def target_depth_fraction(self) -> float:
        return self.valid_target_depth_points / max(self.in_bounds_points, 1)

    @property
    def visible_fraction(self) -> float:
        return self.visible_points / max(self.total_points, 1)


def _check_grid(camera: Camera, shape: tuple[int, int], *, name: str) -> None:
    expected = (camera.intrinsics.height, camera.intrinsics.width)
    if shape != expected:
        raise MarbleInpaintError(
            f"{name} grid {shape} does not match camera grid {expected}"
        )


def lift_edit_region(
    edit_region: np.ndarray,
    depth: np.ndarray,
    camera: Camera,
    *,
    max_points: int,
    min_valid_pixels: int,
    seed: int,
) -> LiftResult:
    """Lift valid edit-region pixel centers into world space."""

    if edit_region.ndim != 2 or depth.ndim != 2:
        raise MarbleInpaintError("edit region and depth must both be HxW arrays")
    if edit_region.shape != depth.shape:
        raise MarbleInpaintError(
            f"edit region shape {edit_region.shape} does not match depth {depth.shape}"
        )
    _check_grid(camera, depth.shape, name="anchor depth")

    edit = edit_region.astype(bool)
    valid = edit & np.isfinite(depth) & (depth > 0)
    edit_pixels = int(edit.sum())
    valid_depth_pixels = int(valid.sum())
    if edit_pixels == 0:
        raise MarbleInpaintError("anchor edit region is empty")
    if valid_depth_pixels < min_valid_pixels:
        raise MarbleInpaintError(
            "anchor edit region has too little valid depth: "
            f"{valid_depth_pixels} pixels; need at least {min_valid_pixels}"
        )

    ys, xs = np.nonzero(valid)
    if valid_depth_pixels > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(valid_depth_pixels, size=max_points, replace=False)
        )
        ys = ys[indices]
        xs = xs[indices]

    z = depth[ys, xs].astype(np.float64)
    intrinsics = camera.intrinsics
    pixel_x = xs.astype(np.float64) + 0.5
    pixel_y = ys.astype(np.float64) + 0.5
    points_camera_rdf = np.column_stack(
        (
            (pixel_x - intrinsics.cx) * z / intrinsics.fx,
            (pixel_y - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    )
    rotation = camera.camera_to_world_rdf()
    position = camera.position_rdf()
    points_world = points_camera_rdf @ rotation.T + position
    return LiftResult(
        points_world=points_world,
        edit_pixels=edit_pixels,
        valid_depth_pixels=valid_depth_pixels,
        sampled_points=len(points_world),
    )


def project_visible_points(
    points_world: np.ndarray,
    camera: Camera,
    depth: np.ndarray,
    *,
    depth_relative_tolerance: float,
) -> ProjectionResult:
    """Project world points and retain those agreeing with target-view depth."""

    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise MarbleInpaintError("points_world must have shape Nx3")
    if depth.ndim != 2:
        raise MarbleInpaintError("target depth must have shape HxW")
    _check_grid(camera, depth.shape, name="target depth")

    rotation = camera.camera_to_world_rdf()
    position = camera.position_rdf()
    points_camera = (points_world - position) @ rotation
    z = points_camera[:, 2]
    intrinsics = camera.intrinsics

    with np.errstate(divide="ignore", invalid="ignore"):
        pixel_x = intrinsics.fx * points_camera[:, 0] / z + intrinsics.cx
        pixel_y = intrinsics.fy * points_camera[:, 1] / z + intrinsics.cy
    in_bounds = (
        np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & np.isfinite(z)
        & (z > 1e-6)
        & (pixel_x >= 0)
        & (pixel_x < intrinsics.width)
        & (pixel_y >= 0)
        & (pixel_y < intrinsics.height)
    )

    safe_x = np.clip(
        np.floor(np.nan_to_num(pixel_x, nan=0.0, posinf=0.0, neginf=0.0)).astype(
            np.int64
        ),
        0,
        intrinsics.width - 1,
    )
    safe_y = np.clip(
        np.floor(np.nan_to_num(pixel_y, nan=0.0, posinf=0.0, neginf=0.0)).astype(
            np.int64
        ),
        0,
        intrinsics.height - 1,
    )
    target_depth = depth[safe_y, safe_x].astype(np.float64)
    valid_target_depth = in_bounds & np.isfinite(target_depth) & (target_depth > 0)
    tolerance = depth_relative_tolerance * np.maximum(target_depth, 1e-6)
    visible = valid_target_depth & (np.abs(target_depth - z) <= tolerance)
    visible_xy = np.column_stack((safe_x[visible], safe_y[visible])).astype(
        np.int32, copy=False
    )
    return ProjectionResult(
        visible_xy=visible_xy,
        total_points=len(points_world),
        in_bounds_points=int(in_bounds.sum()),
        valid_target_depth_points=int(valid_target_depth.sum()),
        visible_points=int(visible.sum()),
    )
