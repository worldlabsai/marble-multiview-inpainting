"""Camera-aware scale-to-cover and center-crop normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.models import Camera, Intrinsics


@dataclass(frozen=True)
class ResizePlan:
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    crop_x: int
    crop_y: int
    output_width: int
    output_height: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sourceWidth": self.source_width,
            "sourceHeight": self.source_height,
            "resizedWidth": self.resized_width,
            "resizedHeight": self.resized_height,
            "cropX": self.crop_x,
            "cropY": self.crop_y,
            "outputWidth": self.output_width,
            "outputHeight": self.output_height,
            "scaleX": self.scale_x,
            "scaleY": self.scale_y,
        }


def scale_to_cover_plan(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
) -> ResizePlan:
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise MarbleInpaintError("image dimensions must be positive")
    scale = max(output_width / source_width, output_height / source_height)
    resized_width = max(output_width, round(source_width * scale))
    resized_height = max(output_height, round(source_height * scale))
    crop_x = (resized_width - output_width) // 2
    crop_y = (resized_height - output_height) // 2
    return ResizePlan(
        source_width=source_width,
        source_height=source_height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_x=crop_x,
        crop_y=crop_y,
        output_width=output_width,
        output_height=output_height,
    )


def _crop_box(plan: ResizePlan) -> tuple[int, int, int, int]:
    return (
        plan.crop_x,
        plan.crop_y,
        plan.crop_x + plan.output_width,
        plan.crop_y + plan.output_height,
    )


def apply_rgb(array: np.ndarray, plan: ResizePlan) -> np.ndarray:
    if array.shape[:2] != (plan.source_height, plan.source_width):
        raise MarbleInpaintError(
            f"RGB shape {array.shape[:2]} does not match resize source "
            f"{(plan.source_height, plan.source_width)}"
        )
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if (plan.resized_width, plan.resized_height) != image.size:
        image = image.resize(
            (plan.resized_width, plan.resized_height), Image.Resampling.LANCZOS
        )
    return np.asarray(image.crop(_crop_box(plan)), dtype=np.uint8)


def apply_mask(array: np.ndarray, plan: ResizePlan) -> np.ndarray:
    if array.shape[:2] != (plan.source_height, plan.source_width):
        raise MarbleInpaintError(
            f"mask shape {array.shape[:2]} does not match resize source "
            f"{(plan.source_height, plan.source_width)}"
        )
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L")
    if (plan.resized_width, plan.resized_height) != image.size:
        image = image.resize(
            (plan.resized_width, plan.resized_height), Image.Resampling.NEAREST
        )
    return np.asarray(image.crop(_crop_box(plan)), dtype=np.uint8)


def apply_depth(array: np.ndarray, plan: ResizePlan) -> np.ndarray:
    if array.shape != (plan.source_height, plan.source_width):
        raise MarbleInpaintError(
            f"depth shape {array.shape} does not match resize source "
            f"{(plan.source_height, plan.source_width)}"
        )
    if (plan.resized_width, plan.resized_height) != (
        plan.source_width,
        plan.source_height,
    ):
        image = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F")
        array = np.asarray(
            image.resize(
                (plan.resized_width, plan.resized_height), Image.Resampling.NEAREST
            ),
            dtype=np.float32,
        )
    y0, x0 = plan.crop_y, plan.crop_x
    return np.asarray(
        array[y0 : y0 + plan.output_height, x0 : x0 + plan.output_width],
        dtype=np.float32,
    )


def update_camera(camera: Camera, plan: ResizePlan) -> Camera:
    source = camera.intrinsics
    if (source.width, source.height) != (plan.source_width, plan.source_height):
        raise MarbleInpaintError(
            "camera intrinsics grid "
            f"{source.width}x{source.height} does not match image grid "
            f"{plan.source_width}x{plan.source_height}"
        )
    intrinsics = Intrinsics(
        width=plan.output_width,
        height=plan.output_height,
        fx=source.fx * plan.scale_x,
        fy=source.fy * plan.scale_y,
        cx=source.cx * plan.scale_x - plan.crop_x,
        cy=source.cy * plan.scale_y - plan.crop_y,
    )
    return Camera(
        position=camera.position,
        quaternion=camera.normalized_quaternion,
        coordinate_system=camera.coordinate_system,
        intrinsics=intrinsics,
    )
