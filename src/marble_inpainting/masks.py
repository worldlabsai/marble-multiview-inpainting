"""Mask rasterization, morphology, polarity conversion, and visualization."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


def disk(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def dilate(region: np.ndarray, radius: int) -> np.ndarray:
    """Dilate by an isotropic pixel radius in time linear in image size."""

    region = np.asarray(region, dtype=bool)
    if radius <= 0 or not region.any():
        return region.copy()
    distance = np.asarray(ndimage.distance_transform_edt(~region), dtype=np.float64)
    return distance <= radius


def rasterize_fill_region(
    visible_xy: np.ndarray,
    *,
    height: int,
    width: int,
    footprint_radius_px: int,
    closing_radius_px: int,
    margin_px: int,
) -> np.ndarray:
    """Turn projected point samples into a connected, conservatively grown mask."""

    footprint = np.zeros((height, width), dtype=bool)
    if len(visible_xy) == 0:
        return footprint
    footprint[visible_xy[:, 1], visible_xy[:, 0]] = True
    if footprint_radius_px:
        footprint = dilate(footprint, footprint_radius_px)
    if closing_radius_px:
        footprint = np.asarray(
            ndimage.binary_closing(footprint, structure=disk(closing_radius_px)),
            dtype=bool,
        )
    footprint = np.asarray(ndimage.binary_fill_holes(footprint), dtype=bool)
    if margin_px:
        footprint = dilate(footprint, margin_px)
    return np.asarray(footprint, dtype=bool)


def keep_mask_from_fill(fill_region: np.ndarray, *, feather_px: int) -> np.ndarray:
    """Convert True=fill into the API convention 0=fill and 255=keep."""

    fill_region = np.asarray(fill_region, dtype=bool)
    if not fill_region.any():
        return np.full(fill_region.shape, 255, dtype=np.uint8)
    if feather_px <= 0:
        return np.where(fill_region, 0, 255).astype(np.uint8)
    distance_outside = np.asarray(
        ndimage.distance_transform_edt(~fill_region), dtype=np.float64
    )
    keep = np.clip(distance_outside / feather_px, 0.0, 1.0) * 255.0
    keep[fill_region] = 0.0
    return np.rint(keep).astype(np.uint8)


def overlay_region(
    rgb: np.ndarray,
    region: np.ndarray,
    *,
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Draw a translucent region and label on an RGB image."""

    base = np.asarray(rgb, dtype=np.uint8).copy()
    region = np.asarray(region, dtype=bool)
    if region.any():
        tint = np.empty_like(base)
        tint[...] = color
        blended = np.rint(base.astype(np.float32) * 0.55 + tint * 0.45).astype(np.uint8)
        base[region] = blended[region]
        boundary = region ^ ndimage.binary_erosion(region, structure=disk(2))
        base[boundary] = np.asarray(color, dtype=np.uint8)

    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    draw.rectangle((6, 6, 14 + text_width, 14 + text_height), fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


def contact_sheet(images: list[np.ndarray], *, columns: int = 3) -> np.ndarray:
    """Create a compact contact sheet from same-shaped RGB images."""

    if not images:
        raise ValueError("contact sheet needs at least one image")
    columns = max(1, min(columns, len(images)))
    source_height, source_width = images[0].shape[:2]
    thumb_width = min(400, source_width)
    thumb_height = max(1, round(source_height * thumb_width / source_width))
    rows = math.ceil(len(images) / columns)
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * thumb_height), color=(24, 24, 24)
    )
    for index, array in enumerate(images):
        image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
        if image.size != (thumb_width, thumb_height):
            image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * thumb_height
        sheet.paste(image, (x, y))
    return np.asarray(sheet, dtype=np.uint8)
