from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from marble_inpainting.masks import keep_mask_from_fill, rasterize_fill_region
from marble_inpainting.scene_io import load_depth, png_header, save_grayscale_png

FIXTURES = Path(__file__).parent / "fixtures"


def _decode_fixture(name: str, destination: Path) -> None:
    encoded = (FIXTURES / f"{name}.b64").read_text(encoding="ascii")
    destination.write_bytes(base64.b64decode(encoded))


def test_keep_mask_polarity_and_feather() -> None:
    fill = np.zeros((9, 9), dtype=bool)
    fill[4, 4] = True
    keep = keep_mask_from_fill(fill, feather_px=2)
    assert keep.dtype == np.uint8
    assert keep[4, 4] == 0
    assert 0 < keep[4, 5] < 255
    assert keep[0, 0] == 255


def test_saved_mask_is_true_8_bit_grayscale_png(tmp_path) -> None:
    path = tmp_path / "mask.png"
    save_grayscale_png(path, np.array([[0, 128, 255]], dtype=np.uint8))
    assert png_header(path) == {
        "width": 3,
        "height": 1,
        "bitDepth": 8,
        "colorType": 0,
    }


def test_rasterization_expands_projected_points() -> None:
    region = rasterize_fill_region(
        np.array([[10, 10]], dtype=np.int32),
        height=21,
        width=21,
        footprint_radius_px=1,
        closing_radius_px=0,
        margin_px=2,
    )
    assert region[10, 10]
    assert region[10, 13]
    assert not region[10, 14]


def test_linear_depth_exr_fixture(tmp_path) -> None:
    expected = np.arange(24, dtype=np.float32).reshape(4, 6) + 1
    path = tmp_path / "depth.exr"
    _decode_fixture("depth.exr", path)
    actual = load_depth(path)
    np.testing.assert_allclose(actual, expected)


def test_multichannel_exr_uses_red_channel(tmp_path) -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    path = tmp_path / "rgb-depth.exr"
    _decode_fixture("rgb-depth.exr", path)
    actual = load_depth(path)
    np.testing.assert_allclose(actual, expected)
