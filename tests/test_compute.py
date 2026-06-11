from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aind_motion_energy.compute import compute_motion_energy


def _info(width=4, height=3, n_frames=5, fps=30.0, bit_depth=8):
    return {"width": width, "height": height, "n_frames": n_frames,
            "fps": fps, "bit_depth": bit_depth, "color_range": "tv"}


def _frames(n, height=3, width=4, fill=None):
    if fill is None:
        return [np.zeros((height, width), dtype=np.uint8) for _ in range(n)]
    return [np.full((height, width), v, dtype=np.uint8) for v in fill]


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_scalar_value_no_normalize(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames(2, H, W, fill=[10, 20]))

    me, _, _ = compute_motion_energy(Path("fake.mp4"), normalize=False)

    assert me.shape == (1,)
    assert me[0] == pytest.approx(10.0 * H * W)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_normalization_divides_by_pixel_count(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames(2, H, W, fill=[0, 10]))

    me, _, _ = compute_motion_energy(Path("fake.mp4"), normalize=True)

    assert me[0] == pytest.approx(10.0)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_output_shapes(mock_info, mock_iter):
    H, W, N = 3, 4, 5
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (H, W), dtype=np.uint8) for _ in range(N)]
    mock_info.return_value = _info(width=W, height=H, n_frames=N)
    mock_iter.return_value = iter(frames)

    me, avg_map, _ = compute_motion_energy(Path("fake.mp4"))

    assert me.shape == (N - 1,)
    assert avg_map.shape == (H, W)
    assert me.dtype == np.float32
    assert avg_map.dtype == np.float32


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_avg_map_is_mean_of_per_pixel_diffs(mock_info, mock_iter):
    H, W = 2, 2
    # 3 frames: 0 → 10 → 20, per-pixel diff is always 10
    mock_info.return_value = _info(width=W, height=H, n_frames=3)
    mock_iter.return_value = iter(_frames(3, H, W, fill=[0, 10, 20]))

    _, avg_map, _ = compute_motion_energy(Path("fake.mp4"), normalize=False)

    np.testing.assert_allclose(avg_map, np.full((H, W), 10.0, dtype=np.float32))


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_metadata_fields(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=3, fps=60.0, bit_depth=8)
    mock_iter.return_value = iter(_frames(3, H, W, fill=[0, 5, 10]))

    _, _, meta = compute_motion_energy(Path("fake.mp4"), normalize=True)

    assert meta["n_frames_decoded"] == 3
    assert meta["n_me_frames"] == 2
    assert meta["fps"] == 60.0
    assert meta["normalized"] is True
    assert meta["roi"] is None
    assert meta["pixel_count"] == H * W
    assert meta["width"] == W
    assert meta["height"] == H


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_roi_adjusts_pixel_count_and_map_shape(mock_info, mock_iter):
    roi = (2, 2, 4, 3)  # x, y, w, h → out 4×3
    roi_w, roi_h = roi[2], roi[3]
    mock_info.return_value = _info(width=10, height=10, n_frames=2)
    frames = [np.zeros((roi_h, roi_w), dtype=np.uint8),
              np.full((roi_h, roi_w), 20, dtype=np.uint8)]
    mock_iter.return_value = iter(frames)

    me, avg_map, meta = compute_motion_energy(Path("fake.mp4"), roi=roi, normalize=True)

    assert meta["pixel_count"] == roi_h * roi_w
    assert meta["roi"] == list(roi)
    assert avg_map.shape == (roi_h, roi_w)
    assert me[0] == pytest.approx(20.0)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_frame_window_stored_in_metadata(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=1000)
    mock_iter.return_value = iter(_frames(3, H, W, fill=[0, 5, 10]))

    _, _, meta = compute_motion_energy(Path("fake.mp4"), start_frame=100, end_frame=200)

    assert meta["start_frame"] == 100
    assert meta["end_frame"] == 200
