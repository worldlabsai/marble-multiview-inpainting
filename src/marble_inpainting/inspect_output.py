"""Independent checks for a prepared output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.pipeline import OUTPUT_MARKER
from marble_inpainting.scene_io import png_header, sha256_file


def inspect_prepared(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not (root / OUTPUT_MARKER).is_file():
        raise MarbleInpaintError(f"not a marble-inpaint output directory: {root}")
    report_path = root / "report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarbleInpaintError(f"could not read {report_path}: {exc}") from exc

    errors: list[str] = []
    checked_views: list[dict[str, Any]] = []
    for view in report.get("views", []):
        view_id = view.get("id", "<unknown>")
        image_info = view.get("image", {})
        mask_info = view.get("keepMask", {})
        image_path = root / image_info.get("path", "")
        mask_path = root / mask_info.get("path", "")
        for label, file_path, expected_hash in (
            ("image", image_path, image_info.get("sha256")),
            ("mask", mask_path, mask_info.get("sha256")),
        ):
            if not file_path.is_file():
                errors.append(f"{view_id}: missing {label}: {file_path}")
            elif expected_hash and sha256_file(file_path) != expected_hash:
                errors.append(f"{view_id}: {label} hash does not match report")
        config = report.get("config", {})
        expected_size = (
            config.get("output_width"),
            config.get("output_height"),
        )
        if image_path.is_file():
            try:
                image_header = png_header(image_path)
                if image_header["bitDepth"] != 8 or image_header["colorType"] != 2:
                    errors.append(
                        f"{view_id}: image must be 8-bit RGB; got "
                        f"bitDepth={image_header['bitDepth']} "
                        f"colorType={image_header['colorType']}"
                    )
                if (image_header["width"], image_header["height"]) != expected_size:
                    errors.append(
                        f"{view_id}: image is {image_header['width']}x"
                        f"{image_header['height']}; expected "
                        f"{expected_size[0]}x{expected_size[1]}"
                    )
            except (MarbleInpaintError, OSError) as exc:
                errors.append(f"{view_id}: {exc}")
        header: dict[str, int] | None = None
        if mask_path.is_file():
            try:
                header = png_header(mask_path)
                if header["bitDepth"] != 8 or header["colorType"] != 0:
                    errors.append(
                        f"{view_id}: mask must be 8-bit grayscale; got "
                        f"bitDepth={header['bitDepth']} colorType={header['colorType']}"
                    )
                if (header["width"], header["height"]) != expected_size:
                    errors.append(
                        f"{view_id}: mask is {header['width']}x{header['height']}; "
                        f"expected {expected_size[0]}x{expected_size[1]}"
                    )
                with Image.open(mask_path) as mask_image:
                    mask_values = np.asarray(mask_image)
                if view.get("status") == "trusted_anchor" and not np.all(
                    mask_values == 255
                ):
                    errors.append(f"{view_id}: trusted anchor mask is not all white")
                fill_fraction = view.get("fillFraction")
                if fill_fraction == 0 and not np.all(mask_values == 255):
                    errors.append(
                        f"{view_id}: zero-fill view contains non-white mask pixels"
                    )
                if (
                    isinstance(fill_fraction, (int, float))
                    and fill_fraction > 0
                    and not np.any(mask_values == 0)
                ):
                    errors.append(
                        f"{view_id}: non-empty fill region has no zero pixels"
                    )
            except (MarbleInpaintError, OSError) as exc:
                errors.append(f"{view_id}: {exc}")
        checked_views.append(
            {
                "id": view_id,
                "status": view.get("status"),
                "fillFraction": view.get("fillFraction"),
                "maskPng": header,
            }
        )

    for required in ("prepared-scene.json", "previews/contact-sheet.png"):
        if not (root / required).is_file():
            errors.append(f"missing required output: {required}")
    request_path = root / "atlas-masked-request.template.json"
    if not request_path.is_file():
        errors.append("missing required output: atlas-masked-request.template.json")
    else:
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            context_frames = request.get("contextFrames")
            target_cameras = request.get("targetCameras")
            if not isinstance(context_frames, list) or not isinstance(
                target_cameras, list
            ):
                errors.append("request template needs contextFrames and targetCameras")
            elif len(context_frames) != len(target_cameras):
                errors.append("request template context/target counts differ")
            else:
                for index, (context, target) in enumerate(
                    zip(context_frames, target_cameras, strict=True)
                ):
                    if not isinstance(context, dict) or context.get("camera") != target:
                        errors.append(
                            f"request template camera pair differs at index {index}"
                        )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"could not read request template: {exc}")
    return {
        "ok": not errors,
        "root": str(root),
        "views": checked_views,
        "warnings": report.get("warnings", []),
        "errors": errors,
    }
