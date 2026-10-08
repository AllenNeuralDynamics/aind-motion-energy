"""Motion-energy computation and post-processing of raw traces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .io import _INTRA_ONLY_CODECS, get_video_info, iter_luma_frames
from .rois import FULL_FRAME_ID, RoiSet


@dataclass(frozen=True, eq=False)
class MotionEnergyResult:
    """Motion energy for one video: a whole-frame trace plus one trace per ROI.

    Attributes
    ----------
    me : numpy.ndarray
        Float32 array of shape ``(n_frames - 1, 1 + K)``, raw values, one row
        per frame-to-frame diff. Column 0 is the whole frame (``"full_frame"``);
        columns ``1..K`` follow `roi_ids`.
    keyframe_mask : numpy.ndarray
        Bool array of shape ``(n_frames - 1,)``, True where the diff crosses
        into or out of a keyframe (contaminated by compression pop). Shared by
        every column, since keyframes are a property of the frame.
    roi_ids : tuple[str, ...]
        Column names of `me`, starting with ``"full_frame"``.
    pixel_counts : numpy.ndarray
        Int64 array of shape ``(1 + K,)``, pixels in each column's region. Used
        for normalization.
    roi_bboxes : numpy.ndarray
        Int32 array of shape ``(1 + K, 4)``, each region's ``(x, y, w, h)``.
    avg_map : numpy.ndarray
        Float32 array of shape ``(H, W)``, whole-frame mean abs difference per
        pixel, computed excluding keyframe-contaminated diffs.
    metadata : dict[str, Any]
        Video properties and processing parameters.
    """

    me: np.ndarray
    keyframe_mask: np.ndarray
    roi_ids: tuple[str, ...]
    pixel_counts: np.ndarray
    roi_bboxes: np.ndarray
    avg_map: np.ndarray
    metadata: dict[str, Any]


def compute_motion_energy(
    video_path: Path,
    rois: RoiSet | None = None,
    normalize: bool = True,
    start_frame: int | None = None,
    end_frame: int | None = None,
    mask_keyframes: bool = True,
) -> MotionEnergyResult:
    """Compute motion energy as the sum of absolute frame-to-frame differences.

    Standard for animal behavior analysis (e.g. Musall et al. 2019). Each frame
    is decoded once and differenced once over the whole frame; the whole-frame
    sum and every ROI's sum are then taken from that same diff, so adding ROIs
    never adds a decode pass. The raw motion energy is never altered — diffs
    contaminated by an H.264/HEVC keyframe "pop" are returned as-is and flagged
    in `keyframe_mask` instead. Use `clean_trace` to produce a NaN'd or
    interpolated version for plotting/regression.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    rois : RoiSet or None, optional
        Binary-mask ROIs drawn on this video's frame (see `load_roi_masks` and
        `RoiSet.from_box`). If None, only the whole-frame trace is computed.
    normalize : bool, optional
        Divide each trace by its region's pixel count, by default True.
    start_frame : int or None, optional
        First frame to process, inclusive.
    end_frame : int or None, optional
        Last frame to process, exclusive.
    mask_keyframes : bool, optional
        Detect and flag diffs contaminated by a keyframe pop, by default True.

    Returns
    -------
    MotionEnergyResult
        Traces, keyframe mask, per-ROI geometry, average motion map and
        metadata.

    Raises
    ------
    ValueError
        If `rois` were drawn on a frame of a different size than the video.
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)
    height, width = info["height"], info["width"]
    if rois is not None:
        rois.check_frame_shape(height, width)

    roi_ids: tuple[str, ...] = (FULL_FRAME_ID, *(rois.ids if rois else ()))
    pixel_counts = np.array([height * width, *(rois.pixel_counts if rois else ())], dtype=np.int64)
    roi_bboxes = np.array(
        [(0, 0, width, height), *(rois.bboxes if rois else ())], dtype=np.int32
    ).reshape(-1, 4)

    # Intra-only codecs have no inter-frame pop — every frame is a keyframe but
    # nothing should be masked.
    do_mask = mask_keyframes and info["codec_name"] not in _INTRA_ONLY_CODECS

    rows: list[np.ndarray] = []
    keyframe_flags: list[bool] = []
    # Exact integer accumulator: 255 per diff overflows uint32 only past ~16.8M frames.
    map_accumulator = np.zeros((height, width), dtype=np.uint32)
    map_pair_count = 0
    prev_frame = None
    prev_is_key = False
    frames_read = 0

    tqdm_total = (end_frame or info["n_frames"]) - (start_frame or 0)

    for frame, is_key in tqdm(
        iter_luma_frames(video_path, start_frame=start_frame, end_frame=end_frame),
        total=tqdm_total,
        desc=video_path.stem,
        unit="frame",
    ):
        frame_i = frame.astype(np.int16)
        if prev_frame is not None:
            # |diff| <= 255, so the uint16 view is exact and adds cleanly to uint32.
            diff = np.abs(frame_i - prev_frame).view(np.uint16)
            row = np.empty(len(roi_ids), dtype=np.int64)
            row[0] = diff.sum(dtype=np.int64)
            if rois is not None:
                row[1:] = rois.reduce(diff)
            rows.append(row)
            # A diff is contaminated if either of its frames is a keyframe:
            # the diff into a keyframe and the diff out of it both pop.
            contaminated = do_mask and (is_key or prev_is_key)
            keyframe_flags.append(contaminated)
            if not contaminated:
                map_accumulator += diff
                map_pair_count += 1
        prev_frame = frame_i
        prev_is_key = is_key
        frames_read += 1

    sums = np.stack(rows) if rows else np.zeros((0, len(roi_ids)), dtype=np.int64)
    me = (sums / pixel_counts if normalize else sums).astype(np.float32)
    keyframe_mask = np.array(keyframe_flags, dtype=bool)
    avg_map = (map_accumulator / max(map_pair_count, 1)).astype(np.float32)

    metadata = {
        "video_path": str(video_path),
        "n_frames_decoded": frames_read,
        "n_me_frames": len(me),
        "n_keyframes_masked": int(keyframe_mask.sum()),
        "fps": info["fps"],
        "codec_name": info["codec_name"],
        "width": width,
        "height": height,
        "bit_depth": info["bit_depth"],
        "n_rois": len(roi_ids) - 1,
        "roi_ids": list(roi_ids),
        "pixel_counts": pixel_counts.tolist(),
        "normalized": normalize,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    return MotionEnergyResult(
        me=me,
        keyframe_mask=keyframe_mask,
        roi_ids=roi_ids,
        pixel_counts=pixel_counts,
        roi_bboxes=roi_bboxes,
        avg_map=avg_map,
        metadata=metadata,
    )


def clean_trace(
    motion_energy: np.ndarray,
    keyframe_mask: np.ndarray,
    method: str = "interpolate",
) -> np.ndarray:
    """Return a copy of `motion_energy` with keyframe-contaminated diffs handled.

    Parameters
    ----------
    motion_energy : numpy.ndarray
        Raw motion-energy trace, shape ``(T,)``, or several traces sharing one
        time axis, shape ``(T, K)``.
    keyframe_mask : numpy.ndarray
        Bool array of shape ``(T,)`` flagging keyframe-contaminated time points
        (rows) in `motion_energy`.
    method : str, optional
        ``"interpolate"`` linearly interpolates across flagged points
        (continuous trace, good for regression); ``"nan"`` replaces them with
        NaN (gaps). By default ``"interpolate"``.

    Returns
    -------
    numpy.ndarray
        Copy of `motion_energy` with flagged entries handled per `method`.
        The input is never modified.

    Raises
    ------
    ValueError
        If `method` is not ``"interpolate"`` or ``"nan"``.
    """
    out = motion_energy.astype(np.float32).copy()
    if not keyframe_mask.any():
        return out

    if method == "nan":
        out[keyframe_mask] = np.nan
        return out
    if method == "interpolate":
        idx = np.arange(len(out))
        good = ~keyframe_mask
        columns = out[:, None] if out.ndim == 1 else out  # views, so writes land in `out`
        for col in columns.T:
            col[keyframe_mask] = np.interp(idx[keyframe_mask], idx[good], col[good])
        return out
    raise ValueError(f"Unknown method: {method!r} (expected 'interpolate' or 'nan')")


def align_to_frames(per_diff: np.ndarray, fill: Any = 0.0) -> np.ndarray:
    """Pad a per-diff array to one entry per video frame.

    Motion energy is computed per frame-to-frame diff, so ``N`` decoded frames
    give ``N - 1`` values. Assigning each diff ``|f[i] - f[i-1]|`` to the later
    frame ``i`` and filling frame 0 makes outputs line up 1:1 with frame
    timestamps and other per-frame data (pose tracking, neural traces).

    Parameters
    ----------
    per_diff : numpy.ndarray
        Array of shape ``(N - 1, ...)``, one entry per diff along axis 0.
    fill : Any, optional
        Value for the first frame, which has no preceding frame, by default
        ``0.0``. Use ``False`` for boolean masks.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, ...)`` with the same dtype as `per_diff`.
    """
    head = np.full((1, *per_diff.shape[1:]), fill, dtype=per_diff.dtype)
    return np.concatenate([head, per_diff], axis=0)
