from __future__ import annotations

import numpy as np
from conftest import make_camera
from scipy.spatial.transform import Rotation

from marble_inpainting.geometry import lift_edit_region, project_visible_points


def test_rdf_identity_round_trip_uses_pixel_centers() -> None:
    camera = make_camera()
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    edit = np.zeros_like(depth, dtype=bool)
    edit[2, 3] = True

    lifted = lift_edit_region(
        edit, depth, camera, max_points=8, min_valid_pixels=1, seed=0
    )
    np.testing.assert_allclose(lifted.points_world, [[-0.25, -0.25, 2.0]])

    projected = project_visible_points(
        lifted.points_world, camera, depth, depth_relative_tolerance=0.01
    )
    np.testing.assert_array_equal(projected.visible_xy, [[3, 2]])
    assert projected.visible_points == 1


def test_rub_identity_is_conjugated_to_rdf_and_round_trips() -> None:
    camera = make_camera(coordinate_system="rub")
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    edit = np.zeros_like(depth, dtype=bool)
    edit[2, 3] = True

    lifted = lift_edit_region(
        edit, depth, camera, max_points=8, min_valid_pixels=1, seed=0
    )
    np.testing.assert_allclose(lifted.points_world, [[-0.25, -0.25, 2.0]])
    projected = project_visible_points(
        lifted.points_world, camera, depth, depth_relative_tolerance=0.01
    )
    np.testing.assert_array_equal(projected.visible_xy, [[3, 2]])


def test_rub_position_is_converted_to_rdf_world_coordinates() -> None:
    camera = make_camera(coordinate_system="rub", position=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(camera.position_rdf(), [1.0, -2.0, -3.0])
    np.testing.assert_allclose(camera.camera_to_world_rdf(), np.eye(3))


def test_rub_rotation_matches_homogeneous_conjugation() -> None:
    quaternion_array = np.array([0.2, 0.4, -0.1, 0.9], dtype=np.float64)
    quaternion_array /= np.linalg.norm(quaternion_array)
    quaternion = (
        float(quaternion_array[0]),
        float(quaternion_array[1]),
        float(quaternion_array[2]),
        float(quaternion_array[3]),
    )
    camera = make_camera(coordinate_system="rub", quaternion=quaternion)
    flip = np.diag([1.0, -1.0, -1.0])
    expected = flip @ Rotation.from_quat(quaternion).as_matrix() @ flip
    np.testing.assert_allclose(camera.camera_to_world_rdf(), expected, atol=1e-12)


def test_equivalent_rub_and_rdf_poses_can_be_mixed() -> None:
    rub = make_camera(coordinate_system="rub", position=(1.0, 2.0, 3.0), cx=3.5, cy=2.5)
    rdf = make_camera(
        coordinate_system="rdf", position=(1.0, -2.0, -3.0), cx=3.5, cy=2.5
    )
    depth = np.full((6, 8), 2.0, dtype=np.float32)
    edit = np.zeros_like(depth, dtype=bool)
    edit[2, 3] = True
    point = lift_edit_region(
        edit, depth, rub, max_points=8, min_valid_pixels=1, seed=0
    ).points_world
    projected = project_visible_points(point, rdf, depth, depth_relative_tolerance=0.01)
    np.testing.assert_array_equal(projected.visible_xy, [[3, 2]])


def test_target_depth_rejects_occluded_point() -> None:
    camera = make_camera(cx=3.5, cy=2.5)
    source_depth = np.full((6, 8), 3.0, dtype=np.float32)
    edit = np.zeros_like(source_depth, dtype=bool)
    edit[2, 3] = True
    point = lift_edit_region(
        edit, source_depth, camera, max_points=8, min_valid_pixels=1, seed=0
    ).points_world

    target_depth = np.full((6, 8), 3.0, dtype=np.float32)
    assert (
        project_visible_points(
            point, camera, target_depth, depth_relative_tolerance=0.01
        ).visible_points
        == 1
    )
    target_depth[2, 3] = 2.0
    occluded = project_visible_points(
        point, camera, target_depth, depth_relative_tolerance=0.01
    )
    assert occluded.in_bounds_points == 1
    assert occluded.valid_target_depth_points == 1
    assert occluded.visible_points == 0


def test_lift_sampling_is_deterministic() -> None:
    camera = make_camera(width=20, height=20, fx=10, fy=10, cx=10, cy=10)
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    edit = np.ones_like(depth, dtype=bool)
    first = lift_edit_region(
        edit, depth, camera, max_points=25, min_valid_pixels=1, seed=91
    )
    second = lift_edit_region(
        edit, depth, camera, max_points=25, min_valid_pixels=1, seed=91
    )
    np.testing.assert_array_equal(first.points_world, second.points_world)
    assert first.sampled_points == 25
