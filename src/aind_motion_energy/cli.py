import argparse
import json
from pathlib import Path

import numpy as np

from .compute import compute_motion_energy

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mj2", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute motion energy from behavior videos"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Video file or directory of videos",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
        help="Region of interest in pixels",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="Skip per-pixel normalization (output raw summed differences)",
    )
    parser.add_argument(
        "--format", choices=["npy", "csv", "both"], default="npy",
        help="Output format (default: npy)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    roi = tuple(args.roi) if args.roi else None

    if args.input.is_file():
        videos = [args.input]
    else:
        videos = sorted(
            p for p in args.input.rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not videos:
        print(f"No videos found in {args.input}")
        return

    for video in videos:
        me, meta = compute_motion_energy(video, roi=roi, normalize=not args.no_normalize)
        stem = video.stem

        if args.format in ("npy", "both"):
            np.save(args.output / f"{stem}_motion_energy.npy", me)

        if args.format in ("csv", "both"):
            import csv
            with open(args.output / f"{stem}_motion_energy.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "motion_energy"])
                writer.writerows(enumerate(me))

        with open(args.output / f"{stem}_me_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{stem}: {len(me)} frames | max={me.max():.4f} | mean={me.mean():.4f}")
