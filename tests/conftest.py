from __future__ import annotations

from marble_inpainting.models import Camera, Intrinsics


def make_camera(
    *,
    width: int = 8,
    height: int = 6,
    fx: float = 4.0,
    fy: float = 4.0,
    cx: float = 4.0,
    cy: float = 3.0,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quaternion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    coordinate_system: str = "rdf",
) -> Camera:
    assert coordinate_system in ("rdf", "rub")
    return Camera(
        position=position,
        quaternion=quaternion,
        coordinate_system=coordinate_system,  # type: ignore[arg-type]
        intrinsics=Intrinsics(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        ),
    )
