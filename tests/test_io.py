from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aind_motion_energy.io import get_video_info, iter_luma_frames


def _fake_probe(fps="500/1"):
    return {"streams": [{"r_frame_rate": fps}]}


def _yuv420p_frame(H, W, y_fill=0):
    """Build a fake yuv420p frame buffer: Y plane filled with y_fill, UV zeros."""
    y = np.full(H * W, y_fill, dtype=np.uint8).tobytes()
    uv = bytes(H * W // 2)  # chroma planes (zeros)
    return y + uv


@patch("aind_motion_energy.io.get_video_range_info")
@patch("aind_motion_energy.io.get_nb_frames")
@patch("aind_motion_energy.io.get_frame_dimensions")
@patch("aind_motion_energy.io.probe")
def test_get_video_info_parses_fields(mock_probe, mock_dims, mock_nb, mock_range):
    mock_probe.return_value = _fake_probe("500/1")
    mock_dims.return_value = (720, 540)
    mock_nb.return_value = 15000
    mock_range.return_value = ("tv", 8)

    info = get_video_info(Path("fake.mp4"))

    assert info["width"] == 720
    assert info["height"] == 540
    assert info["n_frames"] == 15000
    assert info["fps"] == pytest.approx(500.0)
    assert info["bit_depth"] == 8


@patch("aind_motion_energy.io.get_video_range_info")
@patch("aind_motion_energy.io.get_nb_frames")
@patch("aind_motion_energy.io.get_frame_dimensions")
@patch("aind_motion_energy.io.probe")
def test_get_video_info_fractional_fps(mock_probe, mock_dims, mock_nb, mock_range):
    mock_probe.return_value = _fake_probe("60000/1001")
    mock_dims.return_value = (1920, 1080)
    mock_nb.return_value = 1000
    mock_range.return_value = ("tv", 8)

    info = get_video_info(Path("fake.mp4"))

    assert abs(info["fps"] - 59.94) < 0.01


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_yields_correct_shape(mock_popen, mock_info):
    H, W = 4, 4
    mock_info.return_value = {"width": W, "height": H, "n_frames": 2,
                              "fps": 30.0, "bit_depth": 8, "color_range": "tv"}

    mock_proc = MagicMock()
    mock_proc.stdout.read.side_effect = [
        _yuv420p_frame(H, W, y_fill=0),
        _yuv420p_frame(H, W, y_fill=1),
        b"",
    ]
    mock_popen.return_value = mock_proc

    frames = list(iter_luma_frames(Path("fake.mp4")))

    assert len(frames) == 2
    assert frames[0].shape == (H, W)
    assert frames[0].dtype == np.uint8
    assert frames[1].sum() == H * W  # Y plane all ones


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_extracts_y_plane_only(mock_popen, mock_info):
    H, W = 4, 4
    mock_info.return_value = {"width": W, "height": H, "n_frames": 1,
                              "fps": 30.0, "bit_depth": 8, "color_range": "tv"}

    # Y plane = 42, UV plane = 255 — result should only contain 42s
    y = np.full(H * W, 42, dtype=np.uint8).tobytes()
    uv = bytes([255] * (H * W // 2))
    mock_proc = MagicMock()
    mock_proc.stdout.read.side_effect = [y + uv, b""]
    mock_popen.return_value = mock_proc

    frames = list(iter_luma_frames(Path("fake.mp4")))

    assert np.all(frames[0] == 42)


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_uses_yuv420p_pix_fmt(mock_popen, mock_info):
    H, W = 4, 4
    mock_info.return_value = {"width": W, "height": H, "n_frames": 1,
                              "fps": 30.0, "bit_depth": 8, "color_range": "tv"}
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    list(iter_luma_frames(Path("fake.mp4")))

    cmd = mock_popen.call_args[0][0]
    assert "-pix_fmt" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-vf" not in cmd  # no color conversion filter without ROI


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_roi_injects_crop_filter(mock_popen, mock_info):
    H, W = 10, 10
    roi = (2, 2, 4, 4)
    roi_w, roi_h = roi[2], roi[3]
    mock_info.return_value = {"width": W, "height": H, "n_frames": 1,
                              "fps": 30.0, "bit_depth": 8, "color_range": "tv"}

    mock_proc = MagicMock()
    mock_proc.stdout.read.side_effect = [_yuv420p_frame(roi_h, roi_w), b""]
    mock_popen.return_value = mock_proc

    frames = list(iter_luma_frames(Path("fake.mp4"), roi=roi))

    assert len(frames) == 1
    assert frames[0].shape == (roi_h, roi_w)
    cmd = mock_popen.call_args[0][0]
    vf_arg = cmd[cmd.index("-vf") + 1]
    assert f"crop={roi_w}:{roi_h}:{roi[0]}:{roi[1]}" in vf_arg


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_start_frame_adds_ss_before_input(mock_popen, mock_info):
    H, W = 4, 4
    mock_info.return_value = {"width": W, "height": H, "n_frames": 100,
                              "fps": 50.0, "bit_depth": 8, "color_range": "tv"}
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    list(iter_luma_frames(Path("fake.mp4"), start_frame=50))

    cmd = mock_popen.call_args[0][0]
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx
    assert float(cmd[ss_idx + 1]) == pytest.approx(1.0)  # 50 frames / 50 fps


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_end_frame_adds_t_after_input(mock_popen, mock_info):
    H, W = 4, 4
    mock_info.return_value = {"width": W, "height": H, "n_frames": 100,
                              "fps": 50.0, "bit_depth": 8, "color_range": "tv"}
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    list(iter_luma_frames(Path("fake.mp4"), start_frame=50, end_frame=100))

    cmd = mock_popen.call_args[0][0]
    i_idx = cmd.index("-i")
    t_idx = cmd.index("-t")
    assert t_idx > i_idx
    assert float(cmd[t_idx + 1]) == pytest.approx(1.0)  # (100-50) frames / 50 fps


@patch("aind_motion_energy.io.get_video_info")
@patch("aind_motion_energy.io.subprocess.Popen")
def test_iter_luma_frames_no_window_has_no_ss_or_t(mock_popen, mock_info):
    H, W = 4, 4
    mock_info.return_value = {"width": W, "height": H, "n_frames": 10,
                              "fps": 30.0, "bit_depth": 8, "color_range": "tv"}
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = b""
    mock_popen.return_value = mock_proc

    list(iter_luma_frames(Path("fake.mp4")))

    cmd = mock_popen.call_args[0][0]
    assert "-ss" not in cmd
    assert "-t" not in cmd
