from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from marble_inpainting.demo import run_demo
from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.inspect_output import inspect_prepared
from marble_inpainting.models import PrepareConfig
from marble_inpainting.pipeline import prepare_manual


def _mask_for_view(prepared: Path, index: int) -> np.ndarray:
    path = sorted((prepared / "masks").glob("*.png"))[index]
    with Image.open(path) as image:
        return np.asarray(image)


def test_synthetic_demo_runs_end_to_end(tmp_path) -> None:
    root = run_demo(tmp_path / "demo")
    prepared = root / "prepared"
    inspection = inspect_prepared(prepared)
    assert inspection["ok"] is True

    left = _mask_for_view(prepared, 0)
    anchor = _mask_for_view(prepared, 1)
    right = _mask_for_view(prepared, 2)
    assert left.min() == 0 and left.max() == 255
    assert np.all(anchor == 255)
    assert right.min() == 0 and right.max() == 255

    report = json.loads((prepared / "report.json").read_text())
    assert [view["status"] for view in report["views"]] == [
        "projected",
        "trusted_anchor",
        "projected",
    ]
    request = json.loads((prepared / "atlas-masked-request.template.json").read_text())
    assert len(request["contextFrames"]) == len(request["targetCameras"]) == 3
    for context, target in zip(
        request["contextFrames"], request["targetCameras"], strict=True
    ):
        assert context["camera"] == target


def test_demo_is_deterministic(tmp_path) -> None:
    first = run_demo(tmp_path / "first") / "prepared"
    second = run_demo(tmp_path / "second") / "prepared"
    for left, right in zip(
        sorted((first / "masks").glob("*.png")),
        sorted((second / "masks").glob("*.png")),
        strict=True,
    ):
        assert left.read_bytes() == right.read_bytes()


def test_inspection_detects_tampered_output(tmp_path) -> None:
    prepared = run_demo(tmp_path / "demo") / "prepared"
    image = sorted((prepared / "images").glob("*.png"))[0]
    image.write_bytes(image.read_bytes() + b"tampered")
    inspection = inspect_prepared(prepared)
    assert inspection["ok"] is False
    assert any("hash does not match" in error for error in inspection["errors"])


def test_overwrite_requires_tool_marker(tmp_path) -> None:
    destination = tmp_path / "not-ours"
    destination.mkdir()
    (destination / "user-file.txt").write_text("keep me")
    with pytest.raises(MarbleInpaintError, match="marker is missing"):
        run_demo(destination, overwrite=True)
    assert (destination / "user-file.txt").read_text() == "keep me"


def test_manual_mode_needs_no_depth(tmp_path) -> None:
    source = tmp_path / "manual-source"
    source.mkdir()
    views = []
    regions: dict[str, Path] = {}
    for index in range(2):
        view_id = f"view_{index}"
        image_path = source / f"{view_id}.png"
        mask_path = source / f"{view_id}-mask.png"
        Image.new("RGB", (8, 6), (20 + index, 30, 40)).save(image_path)
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[2:4, 3:5] = 255
        Image.fromarray(mask, mode="L").save(mask_path)
        regions[view_id] = mask_path
        views.append(
            {
                "id": view_id,
                "image": image_path.name,
                "camera": {
                    "extrinsics": {
                        "position": [float(index), 0, 0],
                        "quaternion": [0, 0, 0, 1],
                        "coordinateSystem": "rdf",
                    },
                    "intrinsics": {
                        "width": 8,
                        "height": 6,
                        "fx": 4,
                        "fy": 4,
                        "cx": 4,
                        "cy": 3,
                    },
                },
            }
        )
    manifest = source / "scene.json"
    manifest.write_text(json.dumps({"schemaVersion": 1, "views": views}))
    edited = source / "edited.png"
    Image.new("RGB", (8, 6), (90, 80, 70)).save(edited)

    output = prepare_manual(
        scene=manifest,
        anchor_view="view_0",
        edited_anchor=edited,
        edit_regions=regions,
        output=tmp_path / "manual-output",
        config=PrepareConfig(
            output_width=8,
            output_height=6,
            margin_px=0,
            feather_px=0,
        ),
    )
    assert np.all(_mask_for_view(output, 0) == 255)
    second = _mask_for_view(output, 1)
    assert np.all(second[2:4, 3:5] == 0)
    assert inspect_prepared(output)["ok"] is True
