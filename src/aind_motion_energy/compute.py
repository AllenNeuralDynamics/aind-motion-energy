from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from .io import _INTRA_ONLY_CODECS, get_video_info, iter_luma_frames


def compute_motion_energy(
    video_path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
    normalize: bool = True,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    mask_keyframes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute motion energy as the sum of absolute frame-to-frame differences.

    Standard for animal behavior analysis (e.g. Musall et al. 2019). The raw
    motion energy is never altered — diffs contaminated by an H.264/HEVC
    keyframe "pop" are returned as-is and flagged in keyframe_mask instead, so
    nothing is discarded. Use clean_trace() to produce a NaN'd or interpolated
    version for plotting/regression.

    Returns (motion_energy, keyframe_mask, avg_map, metadata):
      - motion_energy: float32 array of shape (n_frames - 1,), raw values
      - keyframe_mask: bool array of shape (n_frames - 1,), True where the diff
        crosses into a keyframe (contaminated by compression pop)
      - avg_map: float32 array of shape (H, W), mean abs difference per pixel,
        computed excluding keyframe-contaminated diffs
      - metadata: dict with video properties and processing parameters
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)

    out_h = roi[3] if roi else info["height"]
    out_w = roi[2] if roi else info["width"]
    pixel_count = out_w * out_h

    # Intra-only codecs have no inter-frame pop — every frame is a keyframe but
    # nothing should be masked.
    is_intra_only = info["codec_name"] in _INTRA_ONLY_CODECS
    do_mask = mask_keyframes and not is_intra_only

    me_values = []
    keyframe_flags = []
    map_accumulator = np.zeros((out_h, out_w), dtype=np.float64)
    map_pair_count = 0
    prev_frame = None
    prev_is_key = False
    frames_read = 0

    tqdm_total = (end_frame or info["n_frames"]) - (start_frame or 0)

    for frame, is_key in tqdm(
        iter_luma_frames(video_path, roi=roi, start_frame=start_frame, end_frame=end_frame),
        total=tqdm_total,
        desc=video_path.stem,
        unit="frame",
    ):
        frame_i = frame.astype(np.int16)
        if prev_frame is not None:
            diff = np.abs(frame_i - prev_frame)
            me_values.append(int(diff.sum(dtype=np.int64)))
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

    me = np.array(me_values, dtype=np.float32)
    keyframe_mask = np.array(keyframe_flags, dtype=bool)
    n_diffs = len(me)

    avg_map = (map_accumulator / max(map_pair_count, 1)).astype(np.float32)

    if normalize:
        me /= pixel_count

    metadata = {
        "video_path": str(video_path),
        "n_frames_decoded": frames_read,
        "n_me_frames": n_diffs,
        "n_keyframes_masked": int(keyframe_mask.sum()),
        "fps": info["fps"],
        "codec_name": info["codec_name"],
        "width": info["width"],
        "height": info["height"],
        "bit_depth": info["bit_depth"],
        "roi": list(roi) if roi else None,
        "normalized": normalize,
        "pixel_count": pixel_count,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }

    return me, keyframe_mask, avg_map, metadata


def clean_trace(
    motion_energy: np.ndarray,
    keyframe_mask: np.ndarray,
    method: str = "interpolate",
) -> np.ndarray:
    """Return a copy of motion_energy with keyframe-contaminated diffs handled.

    method="interpolate" linearly interpolates across flagged points (continuous
    trace, good for regression); method="nan" replaces them with NaN (gaps).
    The input is never modified.
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
        out[keyframe_mask] = np.interp(idx[keyframe_mask], idx[good], out[good])
        return out
    raise ValueError(f"Unknown method: {method!r} (expected 'interpolate' or 'nan')")
