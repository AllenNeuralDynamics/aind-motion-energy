"""Motion-energy computation and post-processing of raw traces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .io import _INTRA_ONLY_CODECS, get_video_info, iter_luma_frames
from .rois import FULL_FRAME_ID, check_rois


def check_frame_range(start_frame: int | None, end_frame: int | None) -> None:
    """Raise if ``[start_frame, end_frame)`` can't select any frames.

    Parameters
    ----------
    start_frame : int or None
        First frame, inclusive. None means 0.
    end_frame : int or None
        Last frame, exclusive. None means the end of the video.

    Raises
    ------
    ValueError
        If either bound is negative, or `end_frame` is not after `start_frame`.
    """
    if start_frame is not None and start_frame < 0:
        raise ValueError(f"start frame must be >= 0 (got {start_frame})")
    if end_frame is not None and end_frame <= (start_frame or 0):
        raise ValueError(f"end frame ({end_frame}) must be after start frame ({start_frame or 0})")


@dataclass(frozen=True, eq=False)
class MotionEnergyResult:
    """Motion energy for one video: a whole-frame trace plus one trace per ROI.

    Traces are frame-aligned: row ``i`` is video frame ``first_frame + i`` and
    holds the difference ``|f[i] - f[i-1]|``. Row 0 has no preceding frame, so
    it is NaN (and False in `keyframe_mask`).

    Attributes
    ----------
    me : numpy.ndarray
        Float32 array of shape ``(n_frames, 1 + K)``, raw values. Column 0 is
        the whole frame (``"full_frame"``); columns ``1..K`` follow `roi_ids`.
    keyframe_mask : numpy.ndarray
        Bool array of shape ``(n_frames,)``, True where the frame's diff crosses
        into or out of a keyframe (contaminated by compression pop). Shared by
        every column, since keyframes are a property of the frame.
    first_frame : int
        Video frame number of row 0.
    roi_ids : tuple[str, ...]
        Column names of `me`, starting with ``"full_frame"``.
    pixel_counts : numpy.ndarray
        Int64 array of shape ``(1 + K,)``, pixels in each column's region (the
        normalization divisor).
    avg_map : numpy.ndarray
        Float32 array of shape ``(H, W)``, whole-frame mean abs difference per
        pixel, computed excluding keyframe-contaminated diffs.
    metadata : dict[str, Any]
        Video properties and processing parameters.
    """

    me: np.ndarray
    keyframe_mask: np.ndarray
    first_frame: int
    roi_ids: tuple[str, ...]
    pixel_counts: np.ndarray
    avg_map: np.ndarray
    metadata: dict[str, Any]


def compute_motion_energy(
    video_path: Path,
    rois: dict[str, np.ndarray] | None = None,
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
    rois : dict[str, numpy.ndarray] or None, optional
        ROI name -> boolean mask of the video's frame shape (see
        `load_roi_masks`). If None, only the whole-frame trace is computed.
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
        Frame-aligned traces, keyframe mask, ROI pixel counts, average motion
        map and metadata.

    Raises
    ------
    ValueError
        If the frame range is invalid (see `check_frame_range`), or an ROI
        mask doesn't fit the video (see `check_rois`).
    """
    check_frame_range(start_frame, end_frame)
    video_path = Path(video_path)
    info = get_video_info(video_path)
    height, width = info["height"], info["width"]
    rois = rois or {}
    check_rois(rois, height, width)
    if height * width * 255 > np.iinfo(np.uint32).max:
        raise ValueError(f"{width}x{height} frames are too large (limit ~16.8 megapixels)")

    # All ROIs are summed in one call per frame: gather every mask's pixels
    # (flat indices, concatenated) and add them up segment by segment.
    segments = [np.flatnonzero(mask) for mask in rois.values()]
    pixel_counts = np.array([height * width, *(len(s) for s in segments)], dtype=np.int64)
    flat_idx = np.concatenate(segments) if segments else np.empty(0, dtype=np.intp)
    offsets = np.concatenate([[0], np.cumsum(pixel_counts[1:])[:-1]]).astype(np.intp)

    # Intra-only codecs have no inter-frame pop — every frame is a keyframe but
    # nothing should be masked.
    do_mask = mask_keyframes and info["codec_name"] not in _INTRA_ONLY_CODECS

    # One row per frame, preallocated from the probed frame count (grown if the
    # probe was short): long 1000 fps sessions have millions of rows.
    n_expected = min(end_frame or info["n_frames"], info["n_frames"]) - (start_frame or 0)
    sums = np.zeros((max(n_expected, 1), len(pixel_counts)), dtype=np.int64)
    keyframe_flags = np.zeros(len(sums), dtype=bool)
    # Exact integer accumulator: 255 per diff overflows uint32 only past ~16.8M frames.
    map_accumulator = np.zeros((height, width), dtype=np.uint32)
    map_pair_count = 0
    prev_frame = None
    prev_is_key = False
    n_frames = 0

    for frame, is_key in tqdm(
        iter_luma_frames(video_path, start_frame=start_frame, end_frame=end_frame),
        total=n_expected,
        desc=video_path.stem,
        unit="frame",
    ):
        frame_i = frame.astype(np.int16)
        if n_frames == len(sums):
            sums, keyframe_flags = _grow(sums), _grow(keyframe_flags)
        if prev_frame is not None:
            # |diff| <= 255, so the uint16 view is exact and adds cleanly to uint32.
            diff = np.abs(frame_i - prev_frame).view(np.uint16)
            sums[n_frames, 0] = diff.sum(dtype=np.int64)
            if segments:
                sums[n_frames, 1:] = np.add.reduceat(
                    diff.ravel()[flat_idx], offsets, dtype=np.uint32
                )
            # A diff is contaminated if either of its frames is a keyframe:
            # the diff into a keyframe and the diff out of it both pop.
            contaminated = do_mask and (is_key or prev_is_key)
            keyframe_flags[n_frames] = contaminated
            if not contaminated:
                map_accumulator += diff
                map_pair_count += 1
        prev_frame = frame_i
        prev_is_key = is_key
        n_frames += 1

    sums = sums[:n_frames]
    me = (sums / pixel_counts if normalize else sums).astype(np.float32)
    me[:1] = np.nan  # frame 0 has no preceding frame
    keyframe_mask = keyframe_flags[:n_frames].copy()
    avg_map = (map_accumulator / max(map_pair_count, 1)).astype(np.float32)

    metadata = {
        "video_path": str(video_path),
        "n_frames_decoded": n_frames,
        "n_keyframes_masked": int(keyframe_mask.sum()),
        "fps": info["fps"],
        "codec_name": info["codec_name"],
        "width": width,
        "height": height,
        "bit_depth": info["bit_depth"],
        "n_rois": len(rois),
        "normalized": normalize,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    return MotionEnergyResult(
        me=me,
        keyframe_mask=keyframe_mask,
        first_frame=start_frame or 0,
        roi_ids=(FULL_FRAME_ID, *rois),
        pixel_counts=pixel_counts,
        avg_map=avg_map,
        metadata=metadata,
    )


def _grow(arr: np.ndarray) -> np.ndarray:
    """Return a copy of `arr` with its first axis doubled (new rows zeroed)."""
    out = np.zeros((2 * len(arr), *arr.shape[1:]), dtype=arr.dtype)
    out[: len(arr)] = arr
    return out


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
        columns = out[:, None] if out.ndim == 1 else out  # views, so writes land in `out`
        for col in columns.T:
            # NaN rows (frame 0) can't anchor the interpolation.
            good = ~keyframe_mask & np.isfinite(col)
            col[keyframe_mask] = np.interp(idx[keyframe_mask], idx[good], col[good])
        return out
    raise ValueError(f"Unknown method: {method!r} (expected 'interpolate' or 'nan')")
