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
    parser.add_argument(
        "--summary-plots", action="store_true",
        help="Also save two static PNG summaries per video (ME timeseries + avg motion map)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Also render an MP4 of the footage with a synced scrolling ME plot",
    )
    parser.add_argument(
        "--viz-fps", type=float, default=60.0,
        help="Playback fps of the visualization video (default: 60)",
    )
    parser.add_argument(
        "--viz-window-seconds", type=float, default=3.0,
        help="Width of the scrolling plot window in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--viz-stride", type=int, default=1,
        help="Render every Nth frame in the visualization (default: 1 = lossless)",
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
        # Key outputs on the camera identity, not the bare file stem.
        # New AIND layout nests each camera as <CameraName>/video.mp4, so every
        # camera shares the stem "video" and would overwrite the others; use the
        # parent folder name instead. Old flat layout (e.g. bottom_camera.avi)
        # already carries the camera in the stem.
        stem = video.parent.name if video.stem == "video" else video.stem

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

        if args.summary_plots:
            from .viz import save_summary_plots

            trace_png, map_png = save_summary_plots(
                args.output, stem, me, me_clean, avg_map, fps=meta["fps"],
            )
            print(f"  saved {trace_png.name}, {map_png.name}")

        if args.visualize:
            from .viz import render_motion_energy_video

            mp4_path = render_motion_energy_video(
                video, me_clean, fps_source=meta["fps"],
                output_path=args.output / f"{stem}_motion_energy.mp4",
                raw_trace=me,
                roi=roi, start_frame=args.start_frame, end_frame=args.end_frame,
                window_seconds=args.viz_window_seconds, out_fps=args.viz_fps,
                stride=args.viz_stride,
            )
            print(f"  rendered {mp4_path}")
