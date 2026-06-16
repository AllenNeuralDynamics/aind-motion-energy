import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aind_motion_energy.viz import render_motion_energy_video


def _frames(n, h=8, w=10):
    """n fake (luma_frame, is_keyframe) tuples."""
    return [
        (np.full((h, w), i % 256, dtype=np.uint8), False)
        for i in range(n)
    ]


def _fake_container(captured):
    """Mock av container/stream that records each encode() call's args."""
    stream = MagicMock()
    stream.encode.side_effect = lambda *a: (captured.append(a), [])[1]
    container = MagicMock()
    container.add_stream.return_value = stream
    return container


def _encoded_frame_count(captured):
    # flush call has no positional args; real frames carry the VideoFrame.
    return sum(1 for a in captured if a)


@patch("aind_motion_energy.viz.av.open")
@patch("aind_motion_energy.viz.iter_luma_frames")
def test_renders_one_output_frame_per_decoded_frame(mock_iter, mock_open, tmp_path):
    mock_iter.return_value = iter(_frames(5))
    captured = []
    mock_open.return_value = _fake_container(captured)
    trace = np.arange(4, dtype=np.float32)  # N-1 entries

    out = render_motion_energy_video(
        Path("fake.mp4"), trace, fps_source=500.0,
        output_path=tmp_path / "viz.mp4", dpi=50,
    )

    assert out == tmp_path / "viz.mp4"
    assert _encoded_frame_count(captured) == 5


@patch("aind_motion_energy.viz.av.open")
@patch("aind_motion_energy.viz.iter_luma_frames")
def test_stride_renders_every_nth_frame(mock_iter, mock_open, tmp_path):
    mock_iter.return_value = iter(_frames(5))
    captured = []
    mock_open.return_value = _fake_container(captured)
    trace = np.arange(4, dtype=np.float32)

    render_motion_energy_video(
        Path("fake.mp4"), trace, fps_source=500.0,
        output_path=tmp_path / "viz.mp4", stride=2, dpi=50,
    )

    # frames j = 0, 2, 4 are rendered
    assert _encoded_frame_count(captured) == 3


@patch("aind_motion_energy.viz.av.open")
@patch("aind_motion_energy.viz.iter_luma_frames")
def test_stream_configured_with_even_dims_and_fps(mock_iter, mock_open, tmp_path):
    mock_iter.return_value = iter(_frames(3))
    captured = []
    container = _fake_container(captured)
    mock_open.return_value = container
    trace = np.arange(2, dtype=np.float32)

    render_motion_energy_video(
        Path("fake.mp4"), trace, fps_source=500.0,
        output_path=tmp_path / "viz.mp4", out_fps=30.0, dpi=50,
    )

    container.add_stream.assert_called_once()
    _, kwargs = container.add_stream.call_args
    assert kwargs.get("rate") == 30
    stream = container.add_stream.return_value
    assert stream.width % 2 == 0 and stream.height % 2 == 0
    assert stream.pix_fmt == "yuv420p"


@patch("aind_motion_energy.viz.av.open")
@patch("aind_motion_energy.viz.iter_luma_frames")
def test_empty_decode_writes_nothing(mock_iter, mock_open, tmp_path):
    mock_iter.return_value = iter([])
    captured = []
    mock_open.return_value = _fake_container(captured)

    render_motion_energy_video(
        Path("fake.mp4"), np.array([], dtype=np.float32), fps_source=500.0,
        output_path=tmp_path / "viz.mp4", dpi=50,
    )

    mock_open.assert_not_called()


@patch("aind_motion_energy.viz.iter_luma_frames")
def test_bad_stride_raises(mock_iter, tmp_path):
    mock_iter.return_value = iter(_frames(2))
    with pytest.raises(ValueError):
        render_motion_energy_video(
            Path("fake.mp4"), np.array([0.0], dtype=np.float32), fps_source=500.0,
            output_path=tmp_path / "viz.mp4", stride=0, dpi=50,
        )


@patch("aind_motion_energy.viz.av.open")
@patch("aind_motion_energy.viz.iter_luma_frames")
def test_raw_trace_does_not_change_frame_count(mock_iter, mock_open, tmp_path):
    mock_iter.return_value = iter(_frames(5))
    captured = []
    mock_open.return_value = _fake_container(captured)
    trace = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    raw = np.array([1.0, 10.0, 10.0, 4.0], dtype=np.float32)  # spikes at indices 1 and 2

    render_motion_energy_video(
        Path("fake.mp4"), trace, fps_source=500.0,
        output_path=tmp_path / "viz.mp4", raw_trace=raw, dpi=50,
    )

    assert _encoded_frame_count(captured) == 5


def test_missing_matplotlib_raises_helpful_error(tmp_path, monkeypatch):
    # Simulate matplotlib not being installed.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match="matplotlib"):
        render_motion_energy_video(
            Path("fake.mp4"), np.array([0.0], dtype=np.float32), fps_source=500.0,
            output_path=tmp_path / "viz.mp4", dpi=50,
        )
