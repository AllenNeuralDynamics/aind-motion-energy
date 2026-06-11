from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aind_motion_energy.io import get_video_info, iter_luma_frames


def _fake_probe(fps="500/1", codec="h264"):
    return {"streams": [{"r_frame_rate": fps, "codec_name": codec}]}


# --- get_video_info -------------------------------------------------------

@patch("aind_motion_energy.io.get_video_range_info")
@patch("aind_motion_energy.io.get_nb_frames")
@patch("aind_motion_energy.io.get_frame_dimensions")
@patch("aind_motion_energy.io.probe")
def test_get_video_info_parses_fields(mock_probe, mock_dims, mock_nb, mock_range):
    mock_probe.return_value = _fake_probe("500/1", codec="h264")
    mock_dims.return_value = (720, 540)
    mock_nb.return_value = 15000
    mock_range.return_value = ("tv", 8)

    info = get_video_info(Path("fake.mp4"))

    assert info["width"] == 720
    assert info["height"] == 540
    assert info["n_frames"] == 15000
    assert info["fps"] == pytest.approx(500.0)
    assert info["bit_depth"] == 8
    assert info["codec_name"] == "h264"


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


# --- iter_luma_frames (PyAV mocked) ---------------------------------------

class _FakePlane(np.ndarray):
    """1-D uint8 ndarray (buffer-protocol capable) with height/line_size attrs."""


def _make_plane(arr2d, line_size=None):
    h, w = arr2d.shape
    line_size = line_size or w
    padded = np.zeros((h, line_size), dtype=np.uint8)
    padded[:, :w] = arr2d
    fp = padded.reshape(-1).view(_FakePlane)
    fp.height = h
    fp.line_size = line_size
    return fp


def _make_frame(arr2d, is_key, line_size=None):
    frame = MagicMock()
    frame.planes = [_make_plane(arr2d, line_size=line_size)]
    frame.width = arr2d.shape[1]
    frame.key_frame = is_key
    return frame


def _fake_container(frames):
    container = MagicMock()
    container.streams.video = [MagicMock()]
    container.decode.return_value = iter(frames)
    return container


@patch("aind_motion_energy.io.av.open")
def test_iter_luma_frames_yields_y_plane_and_keyflag(mock_open):
    H, W = 3, 4
    a = np.full((H, W), 5, dtype=np.uint8)
    b = np.full((H, W), 9, dtype=np.uint8)
    mock_open.return_value = _fake_container([_make_frame(a, False), _make_frame(b, True)])

    out = list(iter_luma_frames(Path("fake.mp4")))

    assert len(out) == 2
    (f0, k0), (f1, k1) = out
    assert f0.shape == (H, W) and f0.dtype == np.uint8
    assert np.all(f0 == 5) and np.all(f1 == 9)
    assert k0 is False and k1 is True


@patch("aind_motion_energy.io.av.open")
def test_iter_luma_frames_respects_line_size_padding(mock_open):
    H, W = 2, 3
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    # decoder pads each row to line_size=8; we must read only the first W columns
    mock_open.return_value = _fake_container([_make_frame(a, False, line_size=8)])

    (frame, _), = list(iter_luma_frames(Path("fake.mp4")))

    assert frame.shape == (H, W)
    np.testing.assert_array_equal(frame, a)


@patch("aind_motion_energy.io.av.open")
def test_iter_luma_frames_roi_crops(mock_open):
    full = np.arange(100, dtype=np.uint8).reshape(10, 10)
    roi = (2, 3, 4, 5)  # x, y, w, h
    mock_open.return_value = _fake_container([_make_frame(full, False)])

    (frame, _), = list(iter_luma_frames(Path("fake.mp4"), roi=roi))

    assert frame.shape == (5, 4)  # (h, w)
    np.testing.assert_array_equal(frame, full[3:8, 2:6])


@patch("aind_motion_energy.io.av.open")
def test_iter_luma_frames_start_end_window(mock_open):
    frames = [_make_frame(np.full((2, 2), i, dtype=np.uint8), False) for i in range(10)]
    mock_open.return_value = _fake_container(frames)

    out = list(iter_luma_frames(Path("fake.mp4"), start_frame=3, end_frame=6))

    # frames 3, 4, 5 (end exclusive)
    assert len(out) == 3
    assert [int(f[0, 0]) for f, _ in out] == [3, 4, 5]
