from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aind_motion_energy.compute import clean_trace, compute_motion_energy


def _info(width=4, height=3, n_frames=5, fps=30.0, bit_depth=8, codec_name="h264"):
    return {"width": width, "height": height, "n_frames": n_frames,
            "fps": fps, "bit_depth": bit_depth, "color_range": "tv",
            "codec_name": codec_name}


def _frames(fills, keys=None, height=3, width=4):
    """Build a list of (frame, is_keyframe) tuples from fill values."""
    if keys is None:
        keys = [False] * len(fills)
    return [
        (np.full((height, width), v, dtype=np.uint8), k)
        for v, k in zip(fills, keys)
    ]


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_scalar_value_no_normalize(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames([10, 20], height=H, width=W))

    me, _, _, _ = compute_motion_energy(Path("fake.mp4"), normalize=False)

    assert me.shape == (1,)
    assert me[0] == pytest.approx(10.0 * H * W)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_normalization_divides_by_pixel_count(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames([0, 10], height=H, width=W))

    me, _, _, _ = compute_motion_energy(Path("fake.mp4"), normalize=True)

    assert me[0] == pytest.approx(10.0)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_output_shapes(mock_info, mock_iter):
    H, W, N = 3, 4, 5
    rng = np.random.default_rng(0)
    frames = [(rng.integers(0, 255, (H, W), dtype=np.uint8), False) for _ in range(N)]
    mock_info.return_value = _info(width=W, height=H, n_frames=N)
    mock_iter.return_value = iter(frames)

    me, mask, avg_map, _ = compute_motion_energy(Path("fake.mp4"))

    assert me.shape == (N - 1,)
    assert mask.shape == (N - 1,)
    assert avg_map.shape == (H, W)
    assert me.dtype == np.float32
    assert mask.dtype == bool
    assert avg_map.dtype == np.float32


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_avg_map_is_mean_of_per_pixel_diffs(mock_info, mock_iter):
    H, W = 2, 2
    mock_info.return_value = _info(width=W, height=H, n_frames=3)
    mock_iter.return_value = iter(_frames([0, 10, 20], height=H, width=W))

    _, _, avg_map, _ = compute_motion_energy(Path("fake.mp4"), normalize=False)

    np.testing.assert_allclose(avg_map, np.full((H, W), 10.0, dtype=np.float32))


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_metadata_fields(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=3, fps=60.0)
    mock_iter.return_value = iter(_frames([0, 5, 10], height=H, width=W))

    _, _, _, meta = compute_motion_energy(Path("fake.mp4"), normalize=True)

    assert meta["n_frames_decoded"] == 3
    assert meta["n_me_frames"] == 2
    assert meta["fps"] == 60.0
    assert meta["codec_name"] == "h264"
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
    frames = [(np.zeros((roi_h, roi_w), dtype=np.uint8), False),
              (np.full((roi_h, roi_w), 20, dtype=np.uint8), False)]
    mock_iter.return_value = iter(frames)

    me, _, avg_map, meta = compute_motion_energy(Path("fake.mp4"), roi=roi, normalize=True)

    assert meta["pixel_count"] == roi_h * roi_w
    assert meta["roi"] == list(roi)
    assert avg_map.shape == (roi_h, roi_w)
    assert me[0] == pytest.approx(20.0)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_frame_window_stored_in_metadata(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=1000)
    mock_iter.return_value = iter(_frames([0, 5, 10], height=H, width=W))

    _, _, _, meta = compute_motion_energy(Path("fake.mp4"), start_frame=100, end_frame=200)

    assert meta["start_frame"] == 100
    assert meta["end_frame"] == 200


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_keyframe_masks_both_adjacent_diffs(mock_info, mock_iter):
    H, W = 2, 2
    # 5 frames, frame index 2 is a keyframe → diffs touching it (me[1], me[2]) masked
    mock_info.return_value = _info(width=W, height=H, n_frames=5, codec_name="h264")
    mock_iter.return_value = iter(
        _frames([0, 10, 20, 30, 40], keys=[False, False, True, False, False], height=H, width=W)
    )

    me, mask, _, meta = compute_motion_energy(Path("fake.mp4"), normalize=False)

    assert me.shape == (4,)
    np.testing.assert_array_equal(mask, [False, True, True, False])
    assert meta["n_keyframes_masked"] == 2
    # raw values are preserved even where masked
    assert not np.isnan(me).any()


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_keyframe_excluded_from_avg_map(mock_info, mock_iter):
    H, W = 2, 2
    # frame 2 keyframe; only me[0] (0->10, diff 10) and me[3] (30->... ) feed the map
    mock_info.return_value = _info(width=W, height=H, n_frames=5, codec_name="h264")
    mock_iter.return_value = iter(
        _frames([0, 10, 20, 30, 36], keys=[False, False, True, False, False], height=H, width=W)
    )

    _, mask, avg_map, _ = compute_motion_energy(Path("fake.mp4"), normalize=False)

    # unmasked diffs: me[0]=10, me[3]=6  → map mean = 8
    np.testing.assert_array_equal(mask, [False, True, True, False])
    np.testing.assert_allclose(avg_map, np.full((H, W), 8.0, dtype=np.float32))


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_intra_only_codec_never_masks(mock_info, mock_iter):
    H, W = 2, 2
    # mjpeg: every frame is a keyframe, but nothing should be masked
    mock_info.return_value = _info(width=W, height=H, n_frames=3, codec_name="mjpeg")
    mock_iter.return_value = iter(
        _frames([0, 10, 20], keys=[True, True, True], height=H, width=W)
    )

    _, mask, _, meta = compute_motion_energy(Path("fake.mp4"))

    assert not mask.any()
    assert meta["n_keyframes_masked"] == 0


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_mask_keyframes_false_disables_masking(mock_info, mock_iter):
    H, W = 2, 2
    mock_info.return_value = _info(width=W, height=H, n_frames=3, codec_name="h264")
    mock_iter.return_value = iter(
        _frames([0, 10, 20], keys=[False, True, False], height=H, width=W)
    )

    _, mask, _, _ = compute_motion_energy(Path("fake.mp4"), mask_keyframes=False)

    assert not mask.any()


def test_clean_trace_interpolate():
    me = np.array([1, 10, 10, 4], dtype=np.float32)
    mask = np.array([False, True, True, False])
    out = clean_trace(me, mask, method="interpolate")
    np.testing.assert_allclose(out, [1, 2, 3, 4])
    # original untouched
    np.testing.assert_allclose(me, [1, 10, 10, 4])


def test_clean_trace_nan():
    me = np.array([1, 10, 10, 4], dtype=np.float32)
    mask = np.array([False, True, True, False])
    out = clean_trace(me, mask, method="nan")
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[0] == 1 and out[3] == 4


def test_clean_trace_no_keyframes_is_identity():
    me = np.array([1, 2, 3], dtype=np.float32)
    mask = np.zeros(3, dtype=bool)
    np.testing.assert_allclose(clean_trace(me, mask), me)


def test_clean_trace_bad_method_raises():
    me = np.array([1.0, 2.0])
    mask = np.array([False, True])
    with pytest.raises(ValueError):
        clean_trace(me, mask, method="bogus")
