"""End-to-end local preparation of Marble atlasMasked inputs."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np

from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.geometry import (
    LiftResult,
    ProjectionResult,
    lift_edit_region,
    project_visible_points,
)
from marble_inpainting.masks import (
    contact_sheet,
    keep_mask_from_fill,
    overlay_region,
    rasterize_fill_region,
)
from marble_inpainting.models import Camera, PrepareConfig, Scene, View
from marble_inpainting.normalize import (
    ResizePlan,
    apply_depth,
    apply_mask,
    apply_rgb,
    scale_to_cover_plan,
    update_camera,
)
from marble_inpainting.scene_io import (
    load_depth,
    load_edit_mask,
    load_rgb,
    load_scene,
    png_header,
    save_grayscale_png,
    save_rgb_png,
    sha256_file,
    write_json,
)

OUTPUT_MARKER = ".marble-inpaint-output"


@dataclass(frozen=True)
class NormalizedView:
    source: View
    rgb: np.ndarray
    depth: np.ndarray | None
    camera: Camera
    resize_plan: ResizePlan


def _tool_version() -> str:
    try:
        return version("marble-multiview-inpainting")
    except PackageNotFoundError:  # pragma: no cover
        return "0.1.0"


def _path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _artifact_stem(index: int, view_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", view_id).strip("-.") or "view"
    return f"{index:03d}-{slug}"


def _display_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


@contextmanager
def _staged_output(destination: Path, *, overwrite: bool) -> Iterator[Path]:
    destination = destination.resolve()
    if destination == destination.parent:
        raise MarbleInpaintError("output directory cannot be a filesystem root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_dir():
        raise MarbleInpaintError(f"output path is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise MarbleInpaintError(
                f"output directory is not empty: {destination}; choose another "
                "directory or pass --overwrite"
            )
        if not (destination / OUTPUT_MARKER).is_file():
            raise MarbleInpaintError(
                f"refusing to overwrite unrecognized directory: {destination}; "
                f"the {OUTPUT_MARKER} marker is missing"
            )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    (staging / OUTPUT_MARKER).write_text("marble-inpaint-v1\n", encoding="utf-8")
    try:
        yield staging
        if destination.exists():
            if any(destination.iterdir()):
                shutil.rmtree(destination)
            else:
                destination.rmdir()
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _normalize_views(scene: Scene, config: PrepareConfig) -> list[NormalizedView]:
    normalized: list[NormalizedView] = []
    for view in scene.views:
        rgb = load_rgb(view.image_path)
        source_height, source_width = rgb.shape[:2]
        if (view.camera.intrinsics.width, view.camera.intrinsics.height) != (
            source_width,
            source_height,
        ):
            raise MarbleInpaintError(
                f"view {view.id!r}: camera grid "
                f"{view.camera.intrinsics.width}x{view.camera.intrinsics.height} "
                f"does not match RGB {source_width}x{source_height}"
            )
        depth = load_depth(view.depth_path) if view.depth_path else None
        if depth is not None and depth.shape != (source_height, source_width):
            raise MarbleInpaintError(
                f"view {view.id!r}: depth shape {depth.shape} does not match RGB "
                f"{(source_height, source_width)}"
            )
        plan = scale_to_cover_plan(
            source_width,
            source_height,
            config.output_width,
            config.output_height,
        )
        normalized.append(
            NormalizedView(
                source=view,
                rgb=apply_rgb(rgb, plan),
                depth=apply_depth(depth, plan) if depth is not None else None,
                camera=update_camera(view.camera, plan),
                resize_plan=plan,
            )
        )
    return normalized


def _load_mask_for_view(path: Path, view: NormalizedView) -> np.ndarray:
    mask = load_edit_mask(path)
    source_shape = (
        view.resize_plan.source_height,
        view.resize_plan.source_width,
    )
    if mask.shape != source_shape:
        raise MarbleInpaintError(
            f"edit-region mask for {view.source.id!r} has shape {mask.shape}; "
            f"expected {source_shape}"
        )
    return apply_mask(mask, view.resize_plan) >= 128


def _load_edited_anchor(path: Path, anchor: NormalizedView) -> np.ndarray:
    edited = load_rgb(path)
    source_shape = (
        anchor.resize_plan.source_height,
        anchor.resize_plan.source_width,
    )
    if edited.shape[:2] != source_shape:
        raise MarbleInpaintError(
            f"edited anchor has shape {edited.shape[:2]}; expected {source_shape}"
        )
    return apply_rgb(edited, anchor.resize_plan)


def _projection_fill_region(
    view: NormalizedView,
    lift: LiftResult,
    config: PrepareConfig,
) -> tuple[np.ndarray, ProjectionResult, str, list[str]]:
    if view.depth is None:
        raise MarbleInpaintError(
            f"view {view.source.id!r} has no depth; add depth or supply a "
            "manual override"
        )
    projection = project_visible_points(
        lift.points_world,
        view.camera,
        view.depth,
        depth_relative_tolerance=config.depth_relative_tolerance,
    )
    warnings: list[str] = []
    shape = (config.output_height, config.output_width)
    if projection.in_bounds_points == 0:
        warnings.append("anchor edit region projects entirely outside this view")
        return np.zeros(shape, dtype=bool), projection, "out_of_view", warnings
    if projection.target_depth_fraction < config.min_target_depth_fraction:
        raise MarbleInpaintError(
            f"view {view.source.id!r}: target depth is valid for only "
            f"{projection.target_depth_fraction:.1%} of in-bounds projected points; "
            f"need {config.min_target_depth_fraction:.1%} or a manual override"
        )
    if projection.visible_points == 0:
        warnings.append(
            "anchor edit region is fully occluded according to target-view depth"
        )
        return np.zeros(shape, dtype=bool), projection, "occluded", warnings
    if (
        projection.visible_points < config.min_visible_points
        or projection.visible_fraction < config.min_visible_fraction
    ):
        raise MarbleInpaintError(
            f"view {view.source.id!r}: only {projection.visible_points} points "
            f"({projection.visible_fraction:.2%}) remain visible; supply a manual "
            "override or adjust documented visibility settings"
        )
    fill = rasterize_fill_region(
        projection.visible_xy,
        height=config.output_height,
        width=config.output_width,
        footprint_radius_px=config.footprint_radius_px,
        closing_radius_px=config.closing_radius_px,
        margin_px=config.margin_px,
    )
    return fill, projection, "projected", warnings


def _input_hashes(
    scene: Scene,
    *,
    edited_anchor: Path,
    masks: dict[str, Path],
) -> list[dict[str, str]]:
    base = scene.manifest_path.parent
    files: dict[Path, str] = {scene.manifest_path: "scene manifest"}
    files[edited_anchor] = "edited anchor"
    for view in scene.views:
        files[view.image_path] = f"RGB for {view.id}"
        if view.depth_path:
            files[view.depth_path] = f"depth for {view.id}"
    for view_id, mask_path in masks.items():
        files[mask_path] = f"edit region for {view_id}"
    return [
        {
            "path": _display_path(path, base),
            "role": role,
            "sha256": sha256_file(path),
        }
        for path, role in files.items()
    ]


def _write_outputs(
    *,
    scene: Scene,
    views: list[NormalizedView],
    anchor_view: str,
    edited_anchor: np.ndarray,
    anchor_edit_region: np.ndarray,
    manual_regions: dict[str, np.ndarray],
    manual_paths: dict[str, Path],
    lift: LiftResult | None,
    config: PrepareConfig,
    prompt: str | None,
    destination: Path,
    overwrite: bool,
    mode: Literal["automatic", "manual"],
    edited_anchor_path: Path,
) -> Path:
    global_warnings: list[str] = []
    view_reports: list[dict[str, Any]] = []
    prepared_views: list[dict[str, Any]] = []
    request_context: list[dict[str, Any]] = []
    request_targets: list[dict[str, Any]] = []
    overlay_images: list[np.ndarray] = []

    with _staged_output(destination, overwrite=overwrite) as output:
        for index, view in enumerate(views):
            stem = _artifact_stem(index, view.source.id)
            image_relative = Path("images") / f"{stem}.png"
            mask_relative = Path("masks") / f"{stem}.keep.png"
            overlay_relative = Path("previews") / f"{stem}.overlay.png"
            is_anchor = view.source.id == anchor_view
            warnings: list[str] = []
            projection_report: dict[str, int | float] | None = None

            if is_anchor:
                output_rgb = edited_anchor
                fill_region = np.zeros(
                    (config.output_height, config.output_width), dtype=bool
                )
                visualization_region = anchor_edit_region
                status = "trusted_anchor"
                region_source = "anchor edit region (visualization only)"
            elif view.source.id in manual_regions:
                output_rgb = view.rgb
                fill_region = manual_regions[view.source.id]
                visualization_region = fill_region
                status = "manual"
                region_source = "manual override"
            else:
                if mode == "manual":
                    raise MarbleInpaintError(
                        f"manual mode is missing an edit region for {view.source.id!r}"
                    )
                if lift is None:  # pragma: no cover - internal invariant
                    raise RuntimeError("automatic mode requires lifted points")
                output_rgb = view.rgb
                fill_region, projection, status, warnings = _projection_fill_region(
                    view, lift, config
                )
                visualization_region = fill_region
                region_source = "depth-checked anchor projection"
                projection_report = {
                    "totalPoints": projection.total_points,
                    "inBoundsPoints": projection.in_bounds_points,
                    "inBoundsFraction": projection.in_bounds_fraction,
                    "validTargetDepthPoints": projection.valid_target_depth_points,
                    "targetDepthFraction": projection.target_depth_fraction,
                    "visiblePoints": projection.visible_points,
                    "visibleFraction": projection.visible_fraction,
                }

            keep_mask = keep_mask_from_fill(fill_region, feather_px=config.feather_px)
            save_rgb_png(output / image_relative, output_rgb)
            save_grayscale_png(output / mask_relative, keep_mask)
            header = png_header(output / mask_relative)
            if header["bitDepth"] != 8 or header["colorType"] != 0:
                raise RuntimeError(
                    f"internal error: {mask_relative} is not 8-bit grayscale"
                )
            label = f"{view.source.id} | {status}"
            overlay = overlay_region(
                output_rgb,
                visualization_region,
                color=(0, 210, 255) if is_anchor else (255, 64, 64),
                label=label,
            )
            save_rgb_png(output / overlay_relative, overlay)
            overlay_images.append(overlay)

            camera_dict = view.camera.to_dict()
            prepared_views.append(
                {
                    "id": view.source.id,
                    "role": "edited_anchor" if is_anchor else "original_context",
                    "image": str(image_relative),
                    "keepMask": str(mask_relative),
                    "camera": camera_dict,
                }
            )
            request_context.append(
                {
                    "imageAsset": {"assetId": f"<upload {image_relative}>"},
                    "maskAsset": {"assetId": f"<upload {mask_relative}>"},
                    "camera": camera_dict,
                }
            )
            request_targets.append(camera_dict)

            fill_fraction = float(fill_region.mean())
            if fill_fraction > 0.65:
                warnings.append(
                    f"fill region covers {fill_fraction:.1%} of the view; "
                    "inspect carefully"
                )
            global_warnings.extend(
                f"{view.source.id}: {warning}" for warning in warnings
            )
            view_report: dict[str, Any] = {
                "id": view.source.id,
                "status": status,
                "regionSource": region_source,
                "fillFraction": fill_fraction,
                "resize": view.resize_plan.to_dict(),
                "image": {
                    "path": str(image_relative),
                    "sha256": sha256_file(output / image_relative),
                },
                "keepMask": {
                    "path": str(mask_relative),
                    "sha256": sha256_file(output / mask_relative),
                    "png": header,
                },
                "overlay": str(overlay_relative),
                "warnings": warnings,
            }
            if projection_report is not None:
                view_report["projection"] = projection_report
            view_reports.append(view_report)

        save_rgb_png(
            output / "previews" / "contact-sheet.png", contact_sheet(overlay_images)
        )
        write_json(
            output / "prepared-scene.json",
            {"schemaVersion": 1, "views": prepared_views},
        )
        write_json(
            output / "atlas-masked-request.template.json",
            {
                "contextFrames": request_context,
                "targetCameras": request_targets,
                "prompt": prompt,
                "enhancePrompt": False,
                "returnDepth": False,
            },
        )
        input_masks = {anchor_view: manual_paths[anchor_view]}
        input_masks.update(
            {
                view_id: path
                for view_id, path in manual_paths.items()
                if view_id != anchor_view
            }
        )
        report: dict[str, Any] = {
            "schemaVersion": 1,
            "tool": {
                "name": "marble-multiview-inpainting",
                "version": _tool_version(),
            },
            "mode": mode,
            "anchorView": anchor_view,
            "config": config.to_dict(),
            "inputs": _input_hashes(
                scene,
                edited_anchor=edited_anchor_path,
                masks=input_masks,
            ),
            "anchor": {
                "editPixels": int(anchor_edit_region.sum()),
                "editFraction": float(anchor_edit_region.mean()),
            },
            "views": view_reports,
            "warnings": global_warnings,
        }
        if lift is not None:
            report["anchor"]["validDepthPixels"] = lift.valid_depth_pixels
            report["anchor"]["sampledPoints"] = lift.sampled_points
        write_json(output / "report.json", report)
    return destination.resolve()


def prepare_auto(
    *,
    scene: str | Path | Scene,
    anchor_view: str,
    edited_anchor: str | Path,
    edit_region: str | Path,
    output: str | Path,
    prompt: str | None = None,
    overrides: Mapping[str, str | Path] | None = None,
    config: PrepareConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare all views by projecting one anchor edit region through RGBD."""

    config = config or PrepareConfig()
    loaded_scene = load_scene(scene) if not isinstance(scene, Scene) else scene
    if not 2 <= len(loaded_scene.views) <= 16:
        raise MarbleInpaintError(
            f"automatic mode needs 2 to 16 views; got {len(loaded_scene.views)}"
        )
    anchor_source = loaded_scene.view(anchor_view)
    normalized = _normalize_views(loaded_scene, config)
    normalized_by_id = {view.source.id: view for view in normalized}
    anchor = normalized_by_id[anchor_source.id]
    edited_path = _path(edited_anchor)
    edit_path = _path(edit_region)
    if not edited_path.is_file():
        raise MarbleInpaintError(f"edited anchor does not exist: {edited_path}")
    if not edit_path.is_file():
        raise MarbleInpaintError(f"edit-region mask does not exist: {edit_path}")
    edited = _load_edited_anchor(edited_path, anchor)
    anchor_region = _load_mask_for_view(edit_path, anchor)
    if anchor.depth is None:
        raise MarbleInpaintError(
            f"anchor view {anchor_view!r} has no depth; automatic mode requires it"
        )
    lift = lift_edit_region(
        anchor_region,
        anchor.depth,
        anchor.camera,
        max_points=config.max_points,
        min_valid_pixels=config.min_anchor_depth_pixels,
        seed=config.seed,
    )

    override_arrays: dict[str, np.ndarray] = {}
    override_paths: dict[str, Path] = {anchor_view: edit_path}
    for view_id, raw_path in (overrides or {}).items():
        if view_id == anchor_view:
            raise MarbleInpaintError(
                "the anchor edit region is provided with --edit-region, not an override"
            )
        if view_id not in normalized_by_id:
            loaded_scene.view(view_id)  # raises a consistent error
        path = _path(raw_path)
        if not path.is_file():
            raise MarbleInpaintError(f"override mask does not exist: {path}")
        override_arrays[view_id] = _load_mask_for_view(path, normalized_by_id[view_id])
        override_paths[view_id] = path

    return _write_outputs(
        scene=loaded_scene,
        views=normalized,
        anchor_view=anchor_view,
        edited_anchor=edited,
        anchor_edit_region=anchor_region,
        manual_regions=override_arrays,
        manual_paths=override_paths,
        lift=lift,
        config=config,
        prompt=prompt,
        destination=_path(output),
        overwrite=overwrite,
        mode="automatic",
        edited_anchor_path=edited_path,
    )


def load_edit_region_map(path: str | Path) -> dict[str, Path]:
    mapping_path = _path(path)
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MarbleInpaintError(f"could not read edit-region map: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise MarbleInpaintError("edit-region map must be a non-empty JSON object")
    result: dict[str, Path] = {}
    for view_id, raw_path in data.items():
        if not isinstance(view_id, str) or not isinstance(raw_path, str):
            raise MarbleInpaintError("edit-region map must contain string path values")
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = mapping_path.parent / resolved
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise MarbleInpaintError(
                f"edit-region mask for {view_id!r} does not exist: {resolved}"
            )
        result[view_id] = resolved
    return result


def prepare_manual(
    *,
    scene: str | Path | Scene,
    anchor_view: str,
    edited_anchor: str | Path,
    edit_regions: Mapping[str, str | Path],
    output: str | Path,
    prompt: str | None = None,
    config: PrepareConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Prepare API masks from explicit edit regions for every view."""

    config = config or PrepareConfig()
    loaded_scene = load_scene(scene) if not isinstance(scene, Scene) else scene
    if not 1 <= len(loaded_scene.views) <= 16:
        raise MarbleInpaintError(
            f"manual mode accepts 1 to 16 views; got {len(loaded_scene.views)}"
        )
    loaded_scene.view(anchor_view)
    normalized = _normalize_views(loaded_scene, config)
    normalized_by_id = {view.source.id: view for view in normalized}
    expected = set(normalized_by_id)
    supplied = set(edit_regions)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unknown: {', '.join(extra)}")
        raise MarbleInpaintError(
            "manual mode needs exactly one edit region per view ("
            + "; ".join(details)
            + ")"
        )

    edited_path = _path(edited_anchor)
    if not edited_path.is_file():
        raise MarbleInpaintError(f"edited anchor does not exist: {edited_path}")
    edited = _load_edited_anchor(edited_path, normalized_by_id[anchor_view])
    region_paths = {view_id: _path(path) for view_id, path in edit_regions.items()}
    regions = {
        view_id: _load_mask_for_view(path, normalized_by_id[view_id])
        for view_id, path in region_paths.items()
    }
    anchor_region = regions.pop(anchor_view)

    return _write_outputs(
        scene=loaded_scene,
        views=normalized,
        anchor_view=anchor_view,
        edited_anchor=edited,
        anchor_edit_region=anchor_region,
        manual_regions=regions,
        manual_paths=region_paths,
        lift=None,
        config=config,
        prompt=prompt,
        destination=_path(output),
        overwrite=overwrite,
        mode="manual",
        edited_anchor_path=edited_path,
    )
