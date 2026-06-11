from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from .io import _INTRA_ONLY_CODECS, get_keyframe_indices, get_video_info, iter_luma_frames


def compute_motion_energy(
    video_path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
    normalize: bool = True,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    mask_keyframes: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute motion energy as the sum of absolute frame-to-frame differences.

    Standard for animal behavior analysis (e.g. Musall et al. 2019).
    Returns (motion_energy, avg_map, metadata):
      - motion_energy: float32 array of shape (n_frames - 1,); NaN at keyframe
        transitions when mask_keyframes=True
      - avg_map: float32 array of shape (H, W), mean absolute difference per
        pixel (keyframe transitions excluded from accumulation)
      - metadata: dict with video properties and processing parameters
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)

    out_h = roi[3] if roi else info["height"]
    out_w = roi[2] if roi else info["width"]
    pixel_count = out_w * out_h

    # Keyframe masking — skip for intra-only codecs (every frame is a keyframe)
    is_intra_only = info["codec_name"] in _INTRA_ONLY_CODECS
    if mask_keyframes and not is_intra_only:
        keyframe_indices = get_keyframe_indices(video_path, info["fps"])
        print(f"[aind-motion-energy] found {len(keyframe_indices)} keyframes to mask")
    else:
        keyframe_indices = frozenset()

    me_values = []
    map_accumulator = np.zeros((out_h, out_w), dtype=np.float64)
    map_pair_count = 0
    prev_frame = None
    frames_read = 0
    frame_offset = start_frame or 0

    tqdm_total = (end_frame or info["n_frames"]) - (start_frame or 0)

    for frame in tqdm(
        iter_luma_frames(video_path, roi=roi, start_frame=start_frame, end_frame=end_frame),
        total=tqdm_total,
        desc=video_path.stem,
        unit="frame",
    ):
        frame_f = frame.astype(np.float32)
        if prev_frame is not None:
            abs_idx = frame_offset + frames_read
            if abs_idx in keyframe_indices:
                me_values.append(np.nan)
            else:
                diff = np.abs(frame_f - prev_frame)
                me_values.append(diff.sum())
                map_accumulator += diff
                map_pair_count += 1
        prev_frame = frame_f
        frames_read += 1

    me = np.array(me_values, dtype=np.float32)
    n_diffs = len(me)
    n_masked = int(np.sum(np.isnan(me)))

    avg_map = (map_accumulator / max(map_pair_count, 1)).astype(np.float32)

    if normalize:
        me /= pixel_count  # NaN stays NaN through division

    metadata = {
        "video_path": str(video_path),
        "n_frames_decoded": frames_read,
        "n_me_frames": n_diffs,
        "n_keyframes_masked": n_masked,
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

    return me, avg_map, metadata
