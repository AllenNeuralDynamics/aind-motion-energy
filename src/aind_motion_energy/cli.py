"""Command-line interface for computing motion energy from behavior videos."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compute import MotionEnergyResult, align_to_frames, clean_trace, compute_motion_energy
from .io import get_video_info
from .rois import FULL_FRAME_ID, RoiSet, load_roi_masks

# Bumped whenever the set or layout of saved files changes.
OUTPUT_VERSION = 2
MASKS_SUFFIX = "_masks.npz"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mj2", ".tif", ".tiff"}


def camera_key(video: Path) -> str:
    """Return the camera identity used to name a video's output files.

    New AIND layout nests each camera as ``<CameraName>/video.mp4``, so every
    camera shares the stem ``video``; use the parent folder name instead. Old
    flat layout (e.g. ``bottom_camera.avi``) already carries the camera in the
    stem.

    Parameters
    ----------
    video : Path
        Path to the video file.

    Returns
    -------
    str
        Prefix for this video's output files.
    """
    return video.parent.name if video.stem == "video" else video.stem


def find_key_collisions(videos: list[Path]) -> dict[str, list[Path]]:
    """Find videos that would write to the same output files.

    Parameters
    ----------
    videos : list[Path]
        Videos to be processed.

    Returns
    -------
    dict[str, list[Path]]
        Camera key -> the two or more videos sharing it (e.g. the same camera
        saved as both ``bottom_camera.avi`` and ``bottom_camera.mp4``). Empty
        if every video has a distinct key.
    """
    by_key: dict[str, list[Path]] = {}
    for video in videos:
        by_key.setdefault(camera_key(video), []).append(video)
    return {key: paths for key, paths in by_key.items() if len(paths) > 1}


def discover_videos(input_path: Path, extensions: set[str]) -> list[Path]:
    """List the videos to process and check their output names are distinct.

    Parameters
    ----------
    input_path : Path
        A single video file (processed as-is), or a directory searched
        recursively.
    extensions : set[str]
        Lower-case extensions with leading dot to include from a directory.

    Returns
    -------
    list[Path]
        Videos to process, sorted; empty if none matched.

    Raises
    ------
    ValueError
        If two videos would write the same output files. Raised up front so a
        later video can't silently overwrite an earlier one's outputs.
    """
    if input_path.is_file():
        return [input_path]
    videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in extensions)

    collisions = find_key_collisions(videos)
    if collisions:
        lines = [f"  {key}: " + ", ".join(map(str, paths)) for key, paths in collisions.items()]
        raise ValueError(
            "multiple videos would write the same output files:\n"
            + "\n".join(lines)
            + "\nTheir outputs would overwrite each other. Narrow --input to a single data "
            + "asset, or pick one copy per camera with --extensions (e.g. --extensions .mp4)."
        )
    return videos


def _normalize_extension(ext: str) -> str:
    """Lower-case an extension and ensure it has a leading dot (``MP4`` -> ``.mp4``)."""
    ext = ext.lower()
    return ext if ext.startswith(".") else f".{ext}"


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``aind-motion-energy`` CLI."""
    parser = argparse.ArgumentParser(description="Compute motion energy from behavior videos")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Video file or directory of videos",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        type=_normalize_extension,
        default=sorted(VIDEO_EXTENSIONS),
        metavar="EXT",
        help="Video extensions to process when --input is a directory, e.g. "
        "'--extensions .mp4' (default: all of " + ", ".join(sorted(VIDEO_EXTENSIONS)) + ")",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory (default: results/)",
    )
    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="One rectangular region of interest in pixels (saved as ROI 'roi')",
    )
    roi_group.add_argument(
        "--roi-masks",
        type=Path,
        metavar="PATH",
        help="ROI annotator mask file(s): a directory holding '*_<camera>_masks.npz' files, "
        "matched to each video by camera name, or a single masks file when --input is one video",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip per-pixel normalization (output raw summed differences)",
    )
    parser.add_argument(
        "--format",
        choices=["npz", "csv", "both"],
        default="npz",
        help="Output format (default: npz)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First frame to process (inclusive, 0-indexed)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to process (exclusive)",
    )
    parser.add_argument(
        "--no-mask-keyframes",
        action="store_true",
        help="Disable detection of H.264/HEVC keyframe-contaminated diffs",
    )
    parser.add_argument(
        "--clean-method",
        choices=["interpolate", "nan"],
        default="interpolate",
        help="How the cleaned trace handles keyframe diffs (default: interpolate)",
    )
    parser.add_argument(
        "--summary-plots",
        action="store_true",
        help="Also save two static PNG summaries per video (ME timeseries + avg motion map)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Also render an MP4 of the footage with a synced scrolling ME plot",
    )
    parser.add_argument(
        "--viz-fps",
        type=float,
        default=60.0,
        help="Playback fps of the visualization video (default: 60)",
    )
    parser.add_argument(
        "--viz-window-seconds",
        type=float,
        default=3.0,
        help="Width of the scrolling plot window in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--viz-rois",
        nargs="+",
        metavar="ID",
        default=None,
        help="ROI ids whose traces scroll in the --visualize video "
        "(default: full_frame plus the first 4 ROIs)",
    )
    parser.add_argument(
        "--viz-stride",
        type=int,
        default=1,
        help="Render every Nth frame in the visualization (default: 1 = lossless)",
    )
    return parser


@dataclass
class RoiSource:
    """The ROIs to apply to one video and where they came from."""

    rois: RoiSet | None = None
    masks_path: Path | None = None
    box: tuple[int, int, int, int] | None = None


def _name_key(name: str) -> str:
    """Compare camera names ignoring case and separators: ``BottomCamera`` == ``bottom_camera``."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def assign_mask_files(mask_dir: Path, keys: list[str]) -> dict[str, list[Path]]:
    """Match ROI annotator mask files to camera keys by filename.

    The annotator writes ``<session>_<camera>_masks.npz``. A file belongs to
    camera ``key`` when its trailing ``_``-separated parts spell ``key``,
    ignoring case and separators — so ``s_BottomCamera_masks.npz`` and
    ``s_bottom_camera_masks.npz`` both match ``bottom_camera`` (the annotator
    names cameras ``BottomCamera`` where older videos are ``bottom_camera.mp4``).
    Only whole parts count: ``camera`` does not match ``s_BottomCamera``. When
    several keys fit — ``camera`` and ``side_camera`` both fit
    ``s_side_camera_masks.npz`` — the longest wins.

    Parameters
    ----------
    mask_dir : Path
        Directory holding the mask files (not searched recursively).
    keys : list[str]
        Camera keys of the videos being processed (see `camera_key`).

    Returns
    -------
    dict[str, list[Path]]
        Camera key -> matching mask files, sorted. Keys with no match are absent.
    """
    by_key: dict[str, list[Path]] = {}
    for path in sorted(mask_dir.glob(f"*{MASKS_SUFFIX}")):
        parts = path.name[: -len(MASKS_SUFFIX)].split("_")
        tails = {_name_key("".join(parts[i:])) for i in range(len(parts))}
        fits = [k for k in keys if _name_key(k) in tails]
        if fits:
            by_key.setdefault(max(fits, key=len), []).append(path)
    return by_key


def _mask_files_by_key(videos: list[Path], masks_path: Path) -> dict[str, list[Path]]:
    """Camera key -> mask files for ``--roi-masks`` (a file for one video, or a directory)."""
    if masks_path.is_file():
        if len(videos) != 1:
            raise ValueError(
                "--roi-masks is a single file but --input has several videos; "
                "pass a directory of '*_<camera>_masks.npz' files instead"
            )
        return {camera_key(videos[0]): [masks_path]}
    files = assign_mask_files(masks_path, [camera_key(v) for v in videos])
    matched = {p for paths in files.values() for p in paths}
    for path in sorted(set(masks_path.glob(f"*{MASKS_SUFFIX}")) - matched):
        print(f"[warn] {path.name} matches no video's camera name; it is not used")
    return files


def resolve_rois(
    videos: list[Path],
    roi_box: tuple[int, int, int, int] | None,
    masks_path: Path | None,
) -> dict[Path, RoiSource]:
    """Load and validate every video's ROIs before any video is processed.

    Parameters
    ----------
    videos : list[Path]
        Videos to be processed.
    roi_box : tuple[int, int, int, int] or None
        ``--roi`` rectangle, applied to every video.
    masks_path : Path or None
        ``--roi-masks``: a directory of per-camera mask files, or one file
        (only valid for a single video).

    Returns
    -------
    dict[Path, RoiSource]
        Each video's ROIs. Videos with no mask file get no ROIs (whole-frame
        trace only), and a warning is printed.

    Raises
    ------
    ValueError
        If a masks file is given for several videos, a camera matches more
        than one mask file, or masks don't fit their video's frame size.
    """
    sources = {video: RoiSource() for video in videos}
    if roi_box is not None:
        for video in videos:
            info = get_video_info(video)
            rois = RoiSet.from_box(roi_box, (info["height"], info["width"]))
            sources[video] = RoiSource(rois=rois, box=roi_box)
        return sources
    if masks_path is None:
        return sources

    files = _mask_files_by_key(videos, masks_path)
    for video in videos:
        key = camera_key(video)
        matches = files.get(key, [])
        if len(matches) > 1:
            raise ValueError(
                f"camera {key!r} matches several mask files: " + ", ".join(map(str, matches))
            )
        if not matches:
            print(
                f"[warn] no '*_{key}{MASKS_SUFFIX}' in {masks_path}; "
                f"{video} gets the whole-frame trace only"
            )
            continue
        rois = load_roi_masks(matches[0])
        info = get_video_info(video)
        try:
            rois.check_frame_shape(info["height"], info["width"])
        except ValueError as e:
            raise ValueError(f"{matches[0]} vs {video}: {e}") from e
        sources[video] = RoiSource(rois=rois, masks_path=matches[0])
    return sources


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    """Write an uncompressed NPZ via a temporary file, so a crash never leaves a partial file.

    Parameters
    ----------
    path : Path
        Destination ``.npz`` path.
    **arrays : numpy.ndarray
        Arrays to store, by name.
    """
    # A plain open() (not mkstemp) so the file gets the usual umask permissions.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)  # type: ignore[arg-type]
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(
    path: Path,
    frame_index: np.ndarray,
    result: MotionEnergyResult,
    me: np.ndarray,
    me_clean: np.ndarray,
    keyframe_mask: np.ndarray,
) -> None:
    header = ["frame_index", "is_keyframe"]
    for roi_id in result.roi_ids:
        header += [roi_id, f"{roi_id}_clean"]
    # Interleave raw/clean columns per ROI to match the header.
    values = np.empty((len(me), 2 * me.shape[1]), dtype=np.float32)
    values[:, 0::2] = me
    values[:, 1::2] = me_clean
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(values):
            writer.writerow([frame_index[i], keyframe_mask[i], *row])


def _nanstat(fn: Any, values: np.ndarray) -> float:
    """`fn` (a NaN-aware reduction) of `values`, or NaN when there are no finite values."""
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else float("nan")


def _roi_metadata(source: RoiSource) -> dict[str, Any]:
    return {
        "roi_box": list(source.box) if source.box else None,
        "roi_masks_path": str(source.masks_path) if source.masks_path else None,
        "roi_masks_sha256": _sha256(source.masks_path) if source.masks_path else None,
    }


def _render_extras(
    video: Path,
    args: argparse.Namespace,
    stem: str,
    result: MotionEnergyResult,
    traces: tuple[np.ndarray, np.ndarray, np.ndarray],
    rois: RoiSet | None,
) -> None:
    me, me_clean, keyframe_mask = traces
    fps = result.metadata["fps"]
    if args.summary_plots:
        from .viz import save_summary_plots

        trace_png, map_png = save_summary_plots(
            args.output,
            stem,
            me,
            me_clean,
            result.avg_map,
            fps=fps,
            roi_ids=result.roi_ids,
            keyframe_mask=keyframe_mask,
            rois=rois,
        )
        print(f"  saved {trace_png.name}, {map_png.name}")

    if args.visualize:
        from .viz import render_motion_energy_video

        mp4_path = render_motion_energy_video(
            video,
            me_clean,
            fps_source=fps,
            output_path=args.output / f"{stem}_motion_energy.mp4",
            raw_trace=me,
            roi_ids=result.roi_ids,
            rois=rois,
            viz_rois=_viz_selection(args.viz_rois, result.roi_ids),
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            window_seconds=args.viz_window_seconds,
            out_fps=args.viz_fps,
            stride=args.viz_stride,
        )
        print(f"  rendered {mp4_path}")


def _viz_selection(requested: list[str] | None, available: tuple[str, ...]) -> list[str] | None:
    """Keep the requested ``--viz-rois`` this video has; None (the default set) if none."""
    if requested is None:
        return None
    kept = [r for r in requested if r in available]
    return kept or None


def check_viz_rois(requested: list[str] | None, sources: dict[Path, RoiSource]) -> None:
    """Raise if a ``--viz-rois`` id exists in no video's ROIs (most likely a typo).

    Parameters
    ----------
    requested : list[str] or None
        ``--viz-rois`` ids.
    sources : dict[Path, RoiSource]
        Every video's ROIs (see `resolve_rois`).

    Raises
    ------
    ValueError
        If any requested id is unknown.
    """
    if not requested:
        return
    known = {FULL_FRAME_ID}
    for source in sources.values():
        known.update(source.rois.ids if source.rois else ())
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"--viz-rois {unknown} not found in any video's ROIs: {sorted(known)}")


def _process_video(video: Path, args: argparse.Namespace, source: RoiSource) -> None:
    """Compute motion energy for one video and write its outputs to ``args.output``."""
    result = compute_motion_energy(
        video,
        rois=source.rois,
        normalize=not args.no_normalize,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        mask_keyframes=not args.no_mask_keyframes,
    )
    me_clean = clean_trace(result.me, result.keyframe_mask, method=args.clean_method)
    stem = camera_key(video)

    # Frame-align at the output edge: one value per decoded frame. Frame 0 has no
    # predecessor, so it's NaN in the traces and False in the keyframe mask.
    me = align_to_frames(result.me)
    me_clean = align_to_frames(me_clean)
    keyframe_mask = align_to_frames(result.keyframe_mask)
    first = args.start_frame or 0
    frame_index = np.arange(first, first + len(me), dtype=np.int64)

    save_npz_atomic(
        args.output / f"{stem}_motion_energy.npz",
        me=me,
        me_clean=me_clean,
        keyframe_mask=keyframe_mask,
        frame_index=frame_index,
        roi_ids=np.array(result.roi_ids, dtype=str),
        pixel_counts=result.pixel_counts,
        roi_bboxes=result.roi_bboxes,
        avg_map=result.avg_map,
    )
    if source.masks_path is not None:
        shutil.copyfile(source.masks_path, args.output / f"{stem}_roi_masks.npz")
    if args.format in ("csv", "both"):
        _write_csv(
            args.output / f"{stem}_motion_energy.csv",
            frame_index,
            result,
            me,
            me_clean,
            keyframe_mask,
        )

    meta = {
        **result.metadata,
        **_roi_metadata(source),
        "output_version": OUTPUT_VERSION,
        "n_output_frames": len(me),
        "first_frame_fill": None,  # NaN; JSON has no NaN literal
    }
    with open(args.output / f"{stem}_me_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    full = me[:, 0]
    print(
        f"{stem}: {len(me)} frames | {len(result.roi_ids) - 1} ROIs | "
        f"{meta['n_keyframes_masked']} keyframes | "
        f"full_frame max={_nanstat(np.nanmax, full):.4f} mean={_nanstat(np.nanmean, full):.4f}"
    )
    _render_extras(video, args, stem, result, (me, me_clean, keyframe_mask), source.rois)


def main() -> None:
    """Parse CLI arguments and compute motion energy for each discovered video.

    Discovers videos under ``--input`` (a single file, or a directory searched
    recursively for ``--extensions``), refuses to run if two videos would write
    the same output files, and loads and checks every video's ROIs
    (``--roi`` or ``--roi-masks``) before processing any of them. For each
    video it then writes ``{stem}_motion_energy.npz`` (whole-frame and per-ROI
    traces, keyframe mask, ROI geometry, average motion map) and
    ``{stem}_me_metadata.json`` to ``--output``, plus a copy of the input
    masks. Optionally also writes CSV output, static summary plots, and a
    rendered visualization video, depending on the flags passed.
    """
    parser = _build_parser()
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    roi_box = tuple(args.roi) if args.roi else None

    try:
        videos = discover_videos(args.input, set(args.extensions))
        if not videos:
            print(f"No videos found in {args.input}")
            return
        sources = resolve_rois(videos, roi_box, args.roi_masks)
        check_viz_rois(args.viz_rois, sources)
    except ValueError as e:
        parser.error(str(e))

    for video in videos:
        _process_video(video, args, sources[video])
