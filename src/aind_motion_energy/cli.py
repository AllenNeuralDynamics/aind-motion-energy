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
    parser.add_argument(
        "--start-frame", type=int, default=None,
        help="First frame to process (inclusive, 0-indexed)",
    )
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="Last frame to process (exclusive)",
    )
    parser.add_argument(
        "--no-mask-keyframes", action="store_true",
        help="Disable NaN masking of H.264/HEVC keyframe transitions",
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
        me, avg_map, meta = compute_motion_energy(
            video, roi=roi, normalize=not args.no_normalize,
            start_frame=args.start_frame, end_frame=args.end_frame,
            mask_keyframes=not args.no_mask_keyframes,
        )
        stem = video.stem

        np.save(args.output / f"{stem}_motion_energy.npy", me)
        np.save(args.output / f"{stem}_motion_energy_map.npy", avg_map)

        if args.format in ("csv", "both"):
            import csv
            with open(args.output / f"{stem}_motion_energy.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "motion_energy"])
                writer.writerows(enumerate(me))

        with open(args.output / f"{stem}_me_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{stem}: {len(me)} frames | max={me.max():.4f} | mean={me.mean():.4f}")
