import subprocess
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
from aind_video_utils import probe, get_nb_frames, get_frame_dimensions, get_video_range_info


def get_video_info(video_path: Path) -> dict:
    p = probe(video_path)
    width, height = get_frame_dimensions(p)
    n_frames = get_nb_frames(p)
    color_range, bit_depth = get_video_range_info(p)
    fps_str = p["streams"][0].get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    return {
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "fps": fps,
        "bit_depth": bit_depth,
        "color_range": color_range,
    }


def iter_luma_frames(
    video_path: Path,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> Generator[np.ndarray, None, None]:
    """Yield uint8 grayscale frames from a video via an ffmpeg rawvideo pipe.

    One subprocess for the full video — much faster than per-frame extraction.
    roi is (x, y, w, h) in pixels.
    """
    info = get_video_info(video_path)

    vf_filters = []
    if roi is not None:
        x, y, w, h = roi
        vf_filters.append(f"crop={w}:{h}:{x}:{y}")
    vf_filters.append("format=gray")

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", ",".join(vf_filters),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-loglevel", "error",
        "pipe:1",
    ]

    out_w = roi[2] if roi else info["width"]
    out_h = roi[3] if roi else info["height"]
    frame_bytes = out_w * out_h

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape(out_h, out_w)
    finally:
        proc.stdout.close()
        proc.wait()
