"""Small, dependency-light models for scenes, cameras, and preparation settings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from marble_inpainting.errors import MarbleInpaintError

CoordinateSystem = Literal["rdf", "rub"]


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise MarbleInpaintError(f"unknown {field} field(s): {', '.join(unknown)}")


def _finite_tuple(value: Any, *, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise MarbleInpaintError(f"{field} must contain exactly {length} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise MarbleInpaintError(f"{field} must contain only numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise MarbleInpaintError(f"{field} must contain only finite numbers")
    return result


def _number(mapping: dict[str, Any], key: str, field: str) -> float:
    if key not in mapping:
        raise MarbleInpaintError(f"missing {field}.{key}")
    try:
        value = float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise MarbleInpaintError(f"{field}.{key} must be a number") from exc
    if not math.isfinite(value):
        raise MarbleInpaintError(f"{field}.{key} must be finite")
    return value


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in top-left-origin pixel coordinates."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, data: Any, *, field: str = "camera.intrinsics") -> Intrinsics:
        if not isinstance(data, dict):
            raise MarbleInpaintError(f"{field} must be an object")
        _reject_unknown(data, {"width", "height", "fx", "fy", "cx", "cy"}, field=field)
        try:
            width = int(data["width"])
            height = int(data["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MarbleInpaintError(
                f"{field}.width and {field}.height must be integers"
            ) from exc
        result = cls(
            width=width,
            height=height,
            fx=_number(data, "fx", field),
            fy=_number(data, "fy", field),
            cx=_number(data, "cx", field),
            cy=_number(data, "cy", field),
        )
        if result.width <= 0 or result.height <= 0:
            raise MarbleInpaintError(f"{field} dimensions must be positive")
        if result.fx <= 0 or result.fy <= 0:
            raise MarbleInpaintError(f"{field} focal lengths must be positive")
        return result

    def to_dict(self) -> dict[str, int | float]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }


@dataclass(frozen=True)
class Camera:
    """Marble camera-to-world pose plus pinhole intrinsics."""

    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    coordinate_system: CoordinateSystem
    intrinsics: Intrinsics

    @classmethod
    def from_dict(cls, data: Any, *, field: str = "camera") -> Camera:
        if not isinstance(data, dict):
            raise MarbleInpaintError(f"{field} must be an object")
        _reject_unknown(data, {"extrinsics", "intrinsics"}, field=field)
        extrinsics = data.get("extrinsics")
        if not isinstance(extrinsics, dict):
            raise MarbleInpaintError(f"{field}.extrinsics must be an object")
        _reject_unknown(
            extrinsics,
            {"position", "quaternion", "coordinateSystem", "coordinate_system"},
            field=f"{field}.extrinsics",
        )
        if "coordinateSystem" in extrinsics and "coordinate_system" in extrinsics:
            raise MarbleInpaintError(
                f"{field}.extrinsics cannot set both coordinateSystem and "
                "coordinate_system"
            )
        coordinate_system = extrinsics.get(
            "coordinateSystem", extrinsics.get("coordinate_system", "rub")
        )
        if coordinate_system not in ("rdf", "rub"):
            raise MarbleInpaintError(
                f"{field}.extrinsics.coordinateSystem must be 'rdf' or 'rub'"
            )
        raw_quaternion = _finite_tuple(
            extrinsics.get("quaternion"),
            length=4,
            field=f"{field}.extrinsics.quaternion",
        )
        quaternion = (
            raw_quaternion[0],
            raw_quaternion[1],
            raw_quaternion[2],
            raw_quaternion[3],
        )
        if sum(component * component for component in quaternion) == 0:
            raise MarbleInpaintError(f"{field}.extrinsics.quaternion must be non-zero")
        raw_position = _finite_tuple(
            extrinsics.get("position"),
            length=3,
            field=f"{field}.extrinsics.position",
        )
        return cls(
            position=(raw_position[0], raw_position[1], raw_position[2]),
            quaternion=quaternion,
            coordinate_system=coordinate_system,
            intrinsics=Intrinsics.from_dict(
                data.get("intrinsics"), field=f"{field}.intrinsics"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "extrinsics": {
                "position": list(self.position),
                "quaternion": list(self.normalized_quaternion),
                "coordinateSystem": self.coordinate_system,
            },
            "intrinsics": self.intrinsics.to_dict(),
        }

    @property
    def normalized_quaternion(self) -> tuple[float, float, float, float]:
        norm = math.sqrt(sum(component * component for component in self.quaternion))
        return tuple(component / norm for component in self.quaternion)  # type: ignore[return-value]

    def camera_to_world_rdf(self) -> np.ndarray:
        """Return the camera-to-world rotation in canonical RDF coordinates.

        Marble converts RUB poses with ``F @ camera_to_world @ F``, where
        ``F = diag(1, -1, -1, 1)``. RDF poses pass through unchanged.
        """

        x, y, z, w = self.normalized_quaternion
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        if self.coordinate_system == "rub":
            flip = np.diag([1.0, -1.0, -1.0])
            rotation = flip @ rotation @ flip
        return rotation

    def position_rdf(self) -> np.ndarray:
        """Return the camera position in the same canonical RDF world frame."""

        position = np.asarray(self.position, dtype=np.float64)
        if self.coordinate_system == "rub":
            position = position * np.array([1.0, -1.0, -1.0])
        return position


@dataclass(frozen=True)
class View:
    id: str
    image_path: Path
    depth_path: Path | None
    depth_representation: str | None
    camera: Camera


@dataclass(frozen=True)
class Scene:
    manifest_path: Path
    views: tuple[View, ...]

    def view(self, view_id: str) -> View:
        for view in self.views:
            if view.id == view_id:
                return view
        choices = ", ".join(view.id for view in self.views)
        raise MarbleInpaintError(
            f"unknown anchor view {view_id!r}; available views: {choices}"
        )


@dataclass(frozen=True)
class PrepareConfig:
    """All geometry and mask choices recorded in the output report."""

    output_width: int = 1280
    output_height: int = 720
    max_points: int = 8192
    min_anchor_depth_pixels: int = 64
    depth_relative_tolerance: float = 0.10
    min_target_depth_fraction: float = 0.50
    min_visible_points: int = 32
    min_visible_fraction: float = 0.01
    footprint_radius_px: int = 4
    closing_radius_px: int = 3
    margin_px: int = 24
    feather_px: int = 8
    seed: int = 0

    def __post_init__(self) -> None:
        integer_positive = {
            "output_width": self.output_width,
            "output_height": self.output_height,
            "max_points": self.max_points,
            "min_anchor_depth_pixels": self.min_anchor_depth_pixels,
        }
        for name, value in integer_positive.items():
            if value <= 0:
                raise MarbleInpaintError(f"{name} must be positive")
        integer_nonnegative = {
            "min_visible_points": self.min_visible_points,
            "footprint_radius_px": self.footprint_radius_px,
            "closing_radius_px": self.closing_radius_px,
            "margin_px": self.margin_px,
            "feather_px": self.feather_px,
            "seed": self.seed,
        }
        for name, value in integer_nonnegative.items():
            if value < 0:
                raise MarbleInpaintError(f"{name} must be non-negative")
        fractions = {
            "depth_relative_tolerance": self.depth_relative_tolerance,
            "min_target_depth_fraction": self.min_target_depth_fraction,
            "min_visible_fraction": self.min_visible_fraction,
        }
        for name, value in fractions.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise MarbleInpaintError(f"{name} must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
