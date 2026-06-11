from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aind_motion_energy.compute import compute_motion_energy


def _info(width=4, height=3, n_frames=5, fps=30.0, bit_depth=8, codec_name="mjpeg"):
    return {"width": width, "height": height, "n_frames": n_frames,
            "fps": fps, "bit_depth": bit_depth, "color_range": "tv",
            "codec_name": codec_name}


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
    roi = (2, 2, 4, 3)
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


@patch("aind_motion_energy.compute.get_keyframe_indices")
@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_keyframe_transition_is_nan(mock_info, mock_iter, mock_kf):
    H, W = 2, 2
    # 4 frames → 3 diffs; frame index 2 is a keyframe → diff[1] should be NaN
    mock_info.return_value = _info(width=W, height=H, n_frames=4, codec_name="h264")
    mock_iter.return_value = iter(_frames(4, H, W, fill=[0, 10, 20, 30]))
    mock_kf.return_value = frozenset({2})

    me, _, meta = compute_motion_energy(Path("fake.mp4"), normalize=False)

    assert me.shape == (3,)
    assert not np.isnan(me[0])  # frame 0→1: normal
    assert np.isnan(me[1])      # frame 1→2: keyframe transition
    assert not np.isnan(me[2])  # frame 2→3: normal
    assert meta["n_keyframes_masked"] == 1


@patch("aind_motion_energy.compute.get_keyframe_indices")
@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_keyframe_excluded_from_avg_map(mock_info, mock_iter, mock_kf):
    H, W = 2, 2
    # frame 0→1: diff=10 everywhere; frame 1→2 is keyframe (NaN); frame 2→3: diff=5
    # avg_map should be mean of only frames 0→1 and 2→3 = (10+5)/2 = 7.5
    mock_info.return_value = _info(width=W, height=H, n_frames=4, codec_name="h264")
    mock_iter.return_value = iter(_frames(4, H, W, fill=[0, 10, 20, 25]))
    mock_kf.return_value = frozenset({2})

    _, avg_map, _ = compute_motion_energy(Path("fake.mp4"), normalize=False)

    np.testing.assert_allclose(avg_map, np.full((H, W), 7.5, dtype=np.float32))


@patch("aind_motion_energy.compute.get_keyframe_indices")
@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_intra_only_codec_skips_keyframe_detection(mock_info, mock_iter, mock_kf):
    H, W = 2, 2
    mock_info.return_value = _info(width=W, height=H, n_frames=3, codec_name="mjpeg")
    mock_iter.return_value = iter(_frames(3, H, W, fill=[0, 10, 20]))

    compute_motion_energy(Path("fake.mp4"))

    mock_kf.assert_not_called()
