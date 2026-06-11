import argparse
import json
from pathlib import Path

import numpy as np

from .compute import clean_trace, compute_motion_energy

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
        help="Disable detection of H.264/HEVC keyframe-contaminated diffs",
    )
    parser.add_argument(
        "--clean-method", choices=["interpolate", "nan"], default="interpolate",
        help="How the cleaned trace handles keyframe diffs (default: interpolate)",
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
        me, keyframe_mask, avg_map, meta = compute_motion_energy(
            video, roi=roi, normalize=not args.no_normalize,
            start_frame=args.start_frame, end_frame=args.end_frame,
            mask_keyframes=not args.no_mask_keyframes,
        )
        me_clean = clean_trace(me, keyframe_mask, method=args.clean_method)
        stem = video.stem

        np.save(args.output / f"{stem}_motion_energy.npy", me)
        np.save(args.output / f"{stem}_motion_energy_clean.npy", me_clean)
        np.save(args.output / f"{stem}_keyframe_mask.npy", keyframe_mask)
        np.save(args.output / f"{stem}_motion_energy_map.npy", avg_map)

        if args.format in ("csv", "both"):
            import csv
            with open(args.output / f"{stem}_motion_energy.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "motion_energy", "motion_energy_clean", "is_keyframe"])
                writer.writerows(zip(range(len(me)), me, me_clean, keyframe_mask))

        with open(args.output / f"{stem}_me_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{stem}: {len(me)} diffs | {meta['n_keyframes_masked']} keyframes | "
              f"max={me.max():.4f} | mean={me.mean():.4f}")
