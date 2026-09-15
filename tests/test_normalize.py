from __future__ import annotations

import numpy as np
from conftest import make_camera

from marble_inpainting.normalize import (
    apply_depth,
    apply_mask,
    scale_to_cover_plan,
    update_camera,
)


def test_scale_to_cover_updates_intrinsics_after_center_crop() -> None:
    plan = scale_to_cover_plan(4, 4, 8, 4)
    assert (plan.resized_width, plan.resized_height) == (8, 8)
    assert (plan.crop_x, plan.crop_y) == (0, 2)
    camera = make_camera(width=4, height=4, fx=2.0, fy=3.0, cx=2.0, cy=2.0)
    resized = update_camera(camera, plan)
    assert resized.intrinsics.to_dict() == {
        "width": 8,
        "height": 4,
        "fx": 4.0,
        "fy": 6.0,
        "cx": 4.0,
        "cy": 2.0,
    }


def test_mask_and_depth_receive_identical_nearest_transform() -> None:
    plan = scale_to_cover_plan(4, 4, 8, 4)
    source = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = apply_mask(source, plan)
    depth = apply_depth(source.astype(np.float32), plan)
    np.testing.assert_array_equal(mask, depth.astype(np.uint8))
    assert mask.shape == (4, 8)
