from __future__ import annotations

import argparse
from pathlib import Path

from .core import FolderResult, fade


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pyfade: MATLAB-aligned FADE fog density evaluation.")
    parser.add_argument("input", type=Path, help="Image path, folder path, or .npy array path.")
    parser.add_argument("--workers", type=int, default=1, help="Image-level worker count. Default: 1.")
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar for folder or batch evaluation.",
    )
    parser.add_argument(
        "--return-map",
        action="store_true",
        help="Return density maps in addition to scores.",
    )
    parser.add_argument(
        "--show-map-shape",
        action="store_true",
        help="For a single image or array, also print the density map shape.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = fade(
        args.input,
        workers=args.workers,
        progress=args.progress,
        return_map=args.return_map or args.show_map_shape,
    )

    if isinstance(result, FolderResult):
        for index, (name, score) in enumerate(result.scores.items(), start=1):
            print(f"{index} / {len(result.scores)} {name} FADE: {score:.12f}")
            if args.return_map and result.density_maps is not None:
                print(f"  map shape: {result.density_maps[name].shape}")
        print(f"Mean FADE: {result.mean_score:.12f}")
        print(f"Min FADE: {result.min_score:.12f}")
        print(f"Max FADE: {result.max_score:.12f}")
        return

    if args.return_map or args.show_map_shape:
        score, density_map = result
        if hasattr(score, "shape"):
            print(f"FADE scores shape: {score.shape}")
            print(f"Density maps shape: {density_map.shape}")
        else:
            print(f"FADE: {float(score):.12f}")
            print(f"Density map shape: {density_map.shape}")
        return

    if hasattr(result, "shape"):
        print(result)
        return

    print(f"FADE: {float(result):.12f}")


if __name__ == "__main__":
    main()
