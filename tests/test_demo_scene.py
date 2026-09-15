"""Validate that the visual demo actually exercises depth and occlusion."""

from __future__ import annotations

import numpy as np

from marble_inpainting.demo_scene import (
    CAMERA_POSITIONS,
    CHAIR,
    COLUMN,
    demo_camera,
    render_scene,
)
from marble_inpainting.geometry import lift_edit_region, project_visible_points


def test_curved_chair_projection_rejects_foreground_column() -> None:
    anchor = demo_camera(CAMERA_POSITIONS[1], width=320, height=180)
    source = render_scene(anchor)
    chair = source.object_ids == CHAIR
    np.testing.assert_array_equal(source.rgb[~chair], source.edited_rgb[~chair])
    assert np.isfinite(source.depth).all() and source.depth.min() > 0
    points = lift_edit_region(
        chair, source.depth, anchor, max_points=10000, min_valid_pixels=64, seed=0
    ).points_world
    # The selected surface spans all three world axes, including substantial depth.
    assert np.all(np.ptp(points, axis=0) > 1.0)

    target = demo_camera(CAMERA_POSITIONS[2], width=320, height=180)
    occluded = render_scene(target)
    unobstructed = render_scene(target, include_column=False)
    actual = project_visible_points(
        points, target, occluded.depth, depth_relative_tolerance=0.02
    )
    clear = project_visible_points(
        points, target, unobstructed.depth, depth_relative_tolerance=0.02
    )
    assert 0 < actual.visible_points < clear.visible_points * 0.8
    ids = occluded.object_ids[actual.visible_xy[:, 1], actual.visible_xy[:, 0]]
    assert not np.any(ids == COLUMN)
