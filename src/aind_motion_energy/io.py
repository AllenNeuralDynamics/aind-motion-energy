"""Video probing and frame decoding with PyAV."""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import av
import numpy as np
from aind_video_utils import get_frame_dimensions, get_nb_frames, get_video_range_info, probe

# Codecs where every frame is intra-coded — no inter-frame keyframe pop, so
# no diffs should be masked even though every frame reports as a keyframe.
_INTRA_ONLY_CODECS = {"mjpeg", "rawvideo", "png", "dpx", "tiff", "ffv1", "huffyuv"}


def get_video_info(video_path: Path) -> dict[str, Any]:
    """Probe a video file for the properties motion-energy computation needs.

    Parameters
    ----------
    video_path : Path
        Path to the video file.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``width``, ``height``, ``n_frames``, ``fps``,
        ``bit_depth``, ``color_range``, and ``codec_name``.
    """
    p = probe(video_path)
    width, height = get_frame_dimensions(p)
    n_frames = get_nb_frames(p)
    color_range, bit_depth = get_video_range_info(p)
    fps_str = p["streams"][0].get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)
    codec_name = p["streams"][0].get("codec_name", "unknown")
    return {
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "fps": fps,
        "bit_depth": bit_depth,
        "color_range": color_range,
        "codec_name": codec_name,
    }


def iter_luma_frames(
    video_path: Path,
    roi: tuple[int, int, int, int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> Generator[tuple[np.ndarray, bool], None, None]:
    """Yield ``(luma_frame, is_keyframe)`` pairs by decoding a video with PyAV.

    Reads the raw Y plane directly (no colorspace conversion or level
    expansion) so values match the stored luminance exactly. ``is_keyframe``
    is the decoder's own per-frame flag, so keyframe transitions are
    identified in the same single decode pass — no separate ffprobe call or
    timestamp rounding.

    Parameters
    ----------
    video_path : Path
        Path to the video file.
    roi : tuple[int, int, int, int] or None, optional
        Region of interest as ``(x, y, w, h)`` in pixels. If None, the full
        frame is yielded.
    start_frame : int or None, optional
        First frame to yield, inclusive. Frames are decoded from the start
        and counted, so seeking is exact.
    end_frame : int or None, optional
        Last frame to yield, exclusive.

    Yields
    ------
    luma_frame : numpy.ndarray
        The Y (luma) plane of the frame, cropped to `roi` if given.
    is_keyframe : bool
        Whether the decoder flagged this frame as a keyframe.
    """
    start = start_frame or 0

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 0
        idx = 0
        for frame in container.decode(stream):
            if end_frame is not None and idx >= end_frame:
                break
            if idx >= start:
                plane = frame.planes[0]
                # Respect stride (line_size may exceed width due to padding).
                y = np.frombuffer(memoryview(plane), np.uint8)
                y = y.reshape(plane.height, plane.line_size)
                y = y[:, : frame.width]
                if roi is not None:
                    x, y0, w, h = roi
                    y = y[y0 : y0 + h, x : x + w]
                yield np.ascontiguousarray(y), bool(frame.key_frame)
            idx += 1
    finally:
        container.close()
