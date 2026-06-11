from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from .io import get_video_info, iter_luma_frames


def compute_motion_energy(
    video_path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
    normalize: bool = True,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute motion energy as the sum of absolute frame-to-frame differences.

    Standard for animal behavior analysis (e.g. Musall et al. 2019).
    Returns (motion_energy, avg_map, metadata):
      - motion_energy: float32 array of shape (n_frames - 1,), one scalar per frame pair
      - avg_map: float32 array of shape (H, W), mean absolute difference per pixel
      - metadata: dict with video properties and processing parameters
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)

    out_h = roi[3] if roi else info["height"]
    out_w = roi[2] if roi else info["width"]
    pixel_count = out_w * out_h

    me_values = []
    map_accumulator = np.zeros((out_h, out_w), dtype=np.float64)
    prev_frame = None
    frames_read = 0

    tqdm_total = (end_frame or info["n_frames"]) - (start_frame or 0)

    for frame in tqdm(
        iter_luma_frames(video_path, roi=roi, start_frame=start_frame, end_frame=end_frame),
        total=tqdm_total,
        desc=video_path.stem,
        unit="frame",
    ):
        frame_f = frame.astype(np.float32)
        if prev_frame is not None:
            diff = np.abs(frame_f - prev_frame)
            me_values.append(diff.sum())
            map_accumulator += diff
        prev_frame = frame_f
        frames_read += 1

    me = np.array(me_values, dtype=np.float32)
    n_diffs = len(me)

    # average map: mean absolute difference at each pixel across all frame pairs
    avg_map = (map_accumulator / n_diffs).astype(np.float32)

    if normalize:
        me /= pixel_count

    metadata = {
        "video_path": str(video_path),
        "n_frames_decoded": frames_read,
        "n_me_frames": n_diffs,
        "fps": info["fps"],
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
