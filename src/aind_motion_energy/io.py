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
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """Yield uint8 Y-plane frames from a video via an ffmpeg rawvideo pipe.

    Outputs yuv420p and slices the Y plane directly — no colorspace conversion
    or level expansion, matching the approach used by aind-video-utils.
    roi is (x, y, w, h) in pixels. start_frame/end_frame use fast input-side
    seeking so the actual start may be at the nearest keyframe.
    """
    info = get_video_info(video_path)

    out_w = roi[2] if roi else info["width"]
    out_h = roi[3] if roi else info["height"]

    vf_filters = []
    if roi is not None:
        x, y, w, h = roi
        vf_filters.append(f"crop={w}:{h}:{x}:{y}")

    cmd = ["ffmpeg"]

    if start_frame is not None:
        cmd += ["-ss", f"{start_frame / info['fps']:.6f}"]

    cmd += ["-i", str(video_path)]

    if end_frame is not None:
        n_start = start_frame or 0
        cmd += ["-t", f"{(end_frame - n_start) / info['fps']:.6f}"]

    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-loglevel", "error",
        "pipe:1",
    ]

    # yuv420p layout: Y plane (w*h bytes) + U plane (w*h/4) + V plane (w*h/4)
    y_bytes = out_w * out_h
    frame_bytes = y_bytes * 3 // 2

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8, count=y_bytes).reshape(out_h, out_w)
    finally:
        proc.stdout.close()
        proc.wait()
