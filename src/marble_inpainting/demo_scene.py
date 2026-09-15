"""Small CPU ray tracer for a reproducible 3D input-preparation example.

Rays have camera-space z=1, so the intersection parameter is camera z-depth,
not Euclidean ray length. RGB, depth, normals, and object IDs share one raster.
The room uses RDF world coordinates: +Y down, floor at Y=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from marble_inpainting.models import Camera, Intrinsics

CHAIR = 1
COLUMN = 2
CAMERA_POSITIONS = ((-3.2, -2.6, -5.4), (0.0, -2.2, -6.0), (3.2, -2.6, -5.4))
LOOK_AT = (0.0, -0.8, 0.3)


@dataclass(frozen=True)
class Ellipsoid:
    center: tuple[float, float, float]
    radii: tuple[float, float, float]
    color: tuple[int, int, int]
    object_id: int


@dataclass(frozen=True)
class RenderedView:
    rgb: np.ndarray
    edited_rgb: np.ndarray
    depth: np.ndarray
    object_ids: np.ndarray


def demo_camera(
    position: tuple[float, float, float], *, width: int = 640, height: int = 360
) -> Camera:
    forward = np.asarray(LOOK_AT) - position
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    quaternion = Rotation.from_matrix(np.column_stack((right, down, forward))).as_quat()
    return Camera(
        position=position,
        quaternion=tuple(float(v) for v in quaternion),  # type: ignore[arg-type]
        coordinate_system="rdf",
        intrinsics=Intrinsics(
            width=width,
            height=height,
            fx=width * 0.88,
            fy=width * 0.88,
            cx=width / 2,
            cy=height / 2,
        ),
    )


def _furniture() -> list[Ellipsoid]:
    upholstery = (205, 91, 53)
    wood = (91, 58, 35)
    return [
        Ellipsoid((0, -0.68, 0.05), (0.93, 0.24, 0.72), upholstery, CHAIR),
        Ellipsoid((0, -1.22, 0.57), (0.90, 0.77, 0.22), upholstery, CHAIR),
        Ellipsoid((-0.87, -0.98, 0.10), (0.20, 0.30, 0.76), upholstery, CHAIR),
        Ellipsoid((0.87, -0.98, 0.10), (0.20, 0.30, 0.76), upholstery, CHAIR),
        *[
            Ellipsoid((x, -0.28, z), (0.065, 0.32, 0.065), wood, 3)
            for x in (-0.65, 0.65)
            for z in (-0.42, 0.52)
        ],
        Ellipsoid((-2.05, -0.30, 1.60), (0.34, 0.36, 0.34), (180, 160, 121), 4),
        *[
            Ellipsoid(center, radii, color, 4)
            for center, radii, color in [
                ((-2.05, -0.93, 1.60), (0.10, 0.68, 0.10), (77, 111, 73)),
                ((-2.30, -0.86, 1.60), (0.34, 0.14, 0.24), (81, 125, 81)),
                ((-1.87, -1.15, 1.64), (0.33, 0.17, 0.22), (101, 142, 82)),
                ((-2.18, -1.42, 1.62), (0.26, 0.20, 0.19), (68, 112, 73)),
            ]
        ],
    ]


def _ellipsoid_hit(
    origin: np.ndarray, rays: np.ndarray, obj: Ellipsoid
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray(obj.radii, dtype=np.float32)
    relative = (origin - obj.center) / radii
    scaled_rays = rays / radii
    a = np.sum(scaled_rays * scaled_rays, axis=-1)
    b = np.sum(relative * scaled_rays, axis=-1)
    c = np.sum(relative * relative, axis=-1) - 1
    discriminant = b * b - a * c
    t = (-b - np.sqrt(np.maximum(discriminant, 0))) / a
    valid = (discriminant >= 0) & (t > 1e-4)
    points = origin + t[..., None] * rays
    normals = (points - obj.center) / (radii * radii)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    return np.where(valid, t, np.inf), normals


def render_scene(camera: Camera, *, include_column: bool = True) -> RenderedView:
    """Render the original scene and a synthetic recolored-chair reference."""
    k = camera.intrinsics
    yy, xx = np.indices((k.height, k.width), dtype=np.float32)
    camera_rays = np.stack(
        ((xx + 0.5 - k.cx) / k.fx, (yy + 0.5 - k.cy) / k.fy, np.ones_like(xx)), axis=-1
    )
    rays = camera_rays @ camera.camera_to_world_rdf().astype(np.float32).T
    origin = np.asarray(camera.position, dtype=np.float32)
    depth = np.full(xx.shape, np.inf, dtype=np.float32)
    normals = np.zeros_like(rays)
    colors = np.zeros_like(rays)
    ids = np.zeros(xx.shape, dtype=np.uint8)

    def put(
        t: np.ndarray, normal: np.ndarray, color: np.ndarray, object_id: int
    ) -> None:
        closer = (t > 1e-4) & (t < depth)
        depth[closer] = t[closer]
        normals[closer] = np.broadcast_to(normal, rays.shape)[closer]
        colors[closer] = np.broadcast_to(color, rays.shape)[closer]
        ids[closer] = object_id

    # Floor, rug, and wall establish perspective and world scale.
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_t = -origin[1] / rays[..., 1]
        floor_points = origin + floor_t[..., None] * rays
        wall_t = (3.0 - origin[2]) / rays[..., 2]
        wall_points = origin + wall_t[..., None] * rays
    floor_color = np.full_like(rays, (221, 210, 191))
    grout = (np.mod(floor_points[..., 0], 0.75) < 0.018) | (
        np.mod(floor_points[..., 2], 0.75) < 0.018
    )
    floor_color[grout] = (180, 172, 158)
    rug = (np.abs(floor_points[..., 0]) < 1.48) & (np.abs(floor_points[..., 2]) < 1.15)
    floor_color[rug] = (184, 189, 168)
    rug_stripe = rug & (np.mod(floor_points[..., 0] + 1.48, 0.15) < 0.018)
    floor_color[rug_stripe] = (166, 175, 156)
    put(floor_t, np.array([0, -1, 0]), floor_color, 0)
    wall_color = np.full_like(rays, (231, 228, 215))
    slats = np.mod(wall_points[..., 0] + 5, 0.5) < 0.018
    wall_color[slats] = (210, 207, 194)
    baseboard = wall_points[..., 1] > -0.08
    wall_color[baseboard] = (150, 139, 116)
    put(wall_t, np.array([0, 0, -1]), wall_color, 0)

    furniture = _furniture()
    for obj in furniture:
        t, normal = _ellipsoid_hit(origin, rays, obj)
        put(t, normal, np.asarray(obj.color), obj.object_id)

    # A vertical cylinder blocks the chair in the right-hand camera.
    if include_column:
        relative = origin - np.array([1.38, 0, -1.70])
        a = rays[..., 0] ** 2 + rays[..., 2] ** 2
        b = relative[0] * rays[..., 0] + relative[2] * rays[..., 2]
        c = relative[0] ** 2 + relative[2] ** 2 - 0.22**2
        discriminant = b * b - a * c
        t = (-b - np.sqrt(np.maximum(discriminant, 0))) / a
        points = origin + t[..., None] * rays
        valid = (discriminant >= 0) & (points[..., 1] <= 0)
        normal = points - np.array([1.38, 0, -1.70])
        normal[..., 1] = 0
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        put(np.where(valid, t, np.inf), normal, np.array([112, 145, 153]), COLUMN)

    light = np.array([-0.45, -0.82, -0.35])
    light /= np.linalg.norm(light)
    lighting = 0.48 + 0.52 * np.maximum(normals @ light, 0)
    points = origin + depth[..., None] * rays
    shadow_origin = points + normals * 0.002
    light_rays = np.broadcast_to(light, rays.shape)
    shadow = np.zeros(xx.shape, dtype=bool)
    for obj in furniture:
        t, _ = _ellipsoid_hit(shadow_origin, light_rays, obj)
        shadow |= np.isfinite(t)
    lighting[shadow] *= 0.73
    rgb = np.clip(colors * lighting[..., None], 0, 255).astype(np.uint8)
    edited = rgb.copy()
    chair = ids == CHAIR
    edited[chair] = np.clip(np.array([44, 147, 148]) * lighting[chair, None], 0, 255)
    return RenderedView(rgb, edited, depth, ids)
