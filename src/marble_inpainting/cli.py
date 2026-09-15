"""Command-line interface for local Marble multiview input preparation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from marble_inpainting import __version__
from marble_inpainting.demo import run_demo
from marble_inpainting.errors import MarbleInpaintError
from marble_inpainting.inspect_output import inspect_prepared
from marble_inpainting.models import PrepareConfig
from marble_inpainting.pipeline import (
    load_edit_region_map,
    prepare_auto,
    prepare_manual,
)


def _add_common_config(parser: argparse.ArgumentParser) -> None:
    defaults = PrepareConfig()
    parser.add_argument("--max-points", type=int, default=defaults.max_points)
    parser.add_argument(
        "--min-anchor-depth-pixels",
        type=int,
        default=defaults.min_anchor_depth_pixels,
    )
    parser.add_argument(
        "--depth-relative-tolerance",
        type=float,
        default=defaults.depth_relative_tolerance,
    )
    parser.add_argument(
        "--min-target-depth-fraction",
        type=float,
        default=defaults.min_target_depth_fraction,
    )
    parser.add_argument(
        "--min-visible-points", type=int, default=defaults.min_visible_points
    )
    parser.add_argument(
        "--min-visible-fraction", type=float, default=defaults.min_visible_fraction
    )
    parser.add_argument(
        "--footprint-radius-px", type=int, default=defaults.footprint_radius_px
    )
    parser.add_argument(
        "--closing-radius-px", type=int, default=defaults.closing_radius_px
    )
    parser.add_argument("--margin-px", type=int, default=defaults.margin_px)
    parser.add_argument("--feather-px", type=int, default=defaults.feather_px)
    parser.add_argument("--seed", type=int, default=defaults.seed)


def _config_from_args(args: argparse.Namespace) -> PrepareConfig:
    defaults = PrepareConfig()
    return replace(
        defaults,
        max_points=args.max_points,
        min_anchor_depth_pixels=args.min_anchor_depth_pixels,
        depth_relative_tolerance=args.depth_relative_tolerance,
        min_target_depth_fraction=args.min_target_depth_fraction,
        min_visible_points=args.min_visible_points,
        min_visible_fraction=args.min_visible_fraction,
        footprint_radius_px=args.footprint_radius_px,
        closing_radius_px=args.closing_radius_px,
        margin_px=args.margin_px,
        feather_px=args.feather_px,
        seed=args.seed,
    )


def _override(value: str) -> tuple[str, Path]:
    view_id, separator, raw_path = value.partition("=")
    if not separator or not view_id or not raw_path:
        raise argparse.ArgumentTypeError("expected VIEW_ID=PATH")
    return view_id, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marble-inpaint",
        description="Prepare consistent multiview inputs for Marble inpainting.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="project one anchor edit region into posed RGBD views"
    )
    prepare.add_argument("--scene", required=True, type=Path)
    prepare.add_argument("--anchor-view", required=True)
    prepare.add_argument("--edited-anchor", required=True, type=Path)
    prepare.add_argument("--edit-region", required=True, type=Path)
    prepare.add_argument("--prompt")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument(
        "--override-edit-region",
        action="append",
        default=[],
        type=_override,
        metavar="VIEW_ID=PATH",
    )
    prepare.add_argument("--overwrite", action="store_true")
    _add_common_config(prepare)

    manual = subparsers.add_parser(
        "prepare-manual", help="prepare explicit edit-region masks without depth"
    )
    manual.add_argument("--scene", required=True, type=Path)
    manual.add_argument("--anchor-view", required=True)
    manual.add_argument("--edited-anchor", required=True, type=Path)
    manual.add_argument("--edit-regions", required=True, type=Path)
    manual.add_argument("--prompt")
    manual.add_argument("--output", required=True, type=Path)
    manual.add_argument("--overwrite", action="store_true")
    _add_common_config(manual)

    inspect_parser = subparsers.add_parser(
        "inspect", help="verify a prepared directory and print its report summary"
    )
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    subparsers.add_parser(
        "config", help="print the complete default preparation configuration"
    )

    demo = subparsers.add_parser(
        "demo", help="generate synthetic RGBD inputs and run the full pipeline"
    )
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--overwrite", action="store_true")
    return parser


def _print_inspection(result: dict[str, object]) -> None:
    print(f"Prepared directory: {result['root']}")
    for view in result["views"]:  # type: ignore[union-attr]
        fraction = view.get("fillFraction")
        fraction_text = f"{fraction:.1%}" if isinstance(fraction, float) else "unknown"
        print(
            f"  {view.get('id')}: {view.get('status')}, fill={fraction_text}, "
            "mask=8-bit grayscale"
        )
    for warning in result["warnings"]:  # type: ignore[union-attr]
        print(f"WARNING: {warning}")
    for error in result["errors"]:  # type: ignore[union-attr]
        print(f"ERROR: {error}", file=sys.stderr)
    print("OK" if result["ok"] else "FAILED")


def run(args: argparse.Namespace) -> int:
    if args.command == "prepare":
        overrides = dict(args.override_edit_region)
        if len(overrides) != len(args.override_edit_region):
            raise MarbleInpaintError("duplicate --override-edit-region view id")
        result = prepare_auto(
            scene=args.scene,
            anchor_view=args.anchor_view,
            edited_anchor=args.edited_anchor,
            edit_region=args.edit_region,
            output=args.output,
            prompt=args.prompt,
            overrides=overrides,
            config=_config_from_args(args),
            overwrite=args.overwrite,
        )
        print(f"Prepared inputs: {result}")
        print(f"Inspect: {result / 'previews' / 'contact-sheet.png'}")
        return 0
    if args.command == "prepare-manual":
        result = prepare_manual(
            scene=args.scene,
            anchor_view=args.anchor_view,
            edited_anchor=args.edited_anchor,
            edit_regions=load_edit_region_map(args.edit_regions),
            output=args.output,
            prompt=args.prompt,
            config=_config_from_args(args),
            overwrite=args.overwrite,
        )
        print(f"Prepared inputs: {result}")
        print(f"Inspect: {result / 'previews' / 'contact-sheet.png'}")
        return 0
    if args.command == "inspect":
        result = inspect_prepared(args.path)
        if args.as_json:
            print(json.dumps(result, indent=2))
        else:
            _print_inspection(result)
        return 0 if result["ok"] else 1
    if args.command == "config":
        print(json.dumps(PrepareConfig().to_dict(), indent=2))
        return 0
    if args.command == "demo":
        result = run_demo(args.output, overwrite=args.overwrite)
        prepared = result / "prepared"
        print(f"Demo inputs: {result / 'source'}")
        print(f"Prepared inputs: {prepared}")
        print(f"Inspect: {prepared / 'previews' / 'contact-sheet.png'}")
        return 0
    raise RuntimeError(f"unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except MarbleInpaintError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    main()
