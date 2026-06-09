from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from .io import get_video_info, iter_luma_frames


def compute_motion_energy(
    video_path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Compute motion energy as the sum of absolute frame-to-frame differences.

    Standard for animal behavior analysis (e.g. Musall et al. 2019).
    Returns (motion_energy, metadata):
      - motion_energy: float32 array of shape (n_frames - 1,)
      - metadata: dict with video properties and processing parameters
    """
    video_path = Path(video_path)
    info = get_video_info(video_path)

    pixel_count = (roi[2] * roi[3]) if roi else (info["width"] * info["height"])
    me_values = []
    prev_frame = None

    for frame in tqdm(
        iter_luma_frames(video_path, roi=roi),
        total=info["n_frames"],
        desc=video_path.stem,
        unit="frame",
    ):
        frame_f = frame.astype(np.float32)
        if prev_frame is not None:
            me_values.append(np.abs(frame_f - prev_frame).sum())
        prev_frame = frame_f

    me = np.array(me_values, dtype=np.float32)

    if normalize:
        me /= pixel_count

    metadata = {
        "video_path": str(video_path),
        "n_frames": info["n_frames"],
        "n_me_frames": len(me),
        "fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "bit_depth": info["bit_depth"],
        "roi": list(roi) if roi else None,
        "normalized": normalize,
        "pixel_count": pixel_count,
    }

    return me, metadata
