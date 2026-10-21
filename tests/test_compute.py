from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from aind_motion_energy.compute import MotionEnergyResult, clean_trace, compute_motion_energy
from aind_motion_energy.rois import box_roi


def _info(width=4, height=3, n_frames=5, fps=30.0, bit_depth=8, codec_name="h264"):
    return {
        "width": width,
        "height": height,
        "n_frames": n_frames,
        "fps": fps,
        "bit_depth": bit_depth,
        "color_range": "tv",
        "codec_name": codec_name,
    }


def _legacy(res: MotionEnergyResult):
    """Full-frame per-diff view of a result: (me, mask, avg_map, meta) without row 0.

    Row 0 (frame 0, no preceding frame) is checked to be NaN / False, then dropped.
    """
    if len(res.me):
        assert np.isnan(res.me[0]).all() and not res.keyframe_mask[0]
    return res.me[1:, 0], res.keyframe_mask[1:], res.avg_map, res.metadata


def _frames(fills, keys=None, height=3, width=4):
    """Build a list of (frame, is_keyframe) tuples from fill values."""
    if keys is None:
        keys = [False] * len(fills)
    return [(np.full((height, width), v, dtype=np.uint8), k) for v, k in zip(fills, keys)]


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_scalar_value_no_normalize(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames([10, 20], height=H, width=W))

    me, _, _, _ = _legacy(compute_motion_energy(Path("fake.mp4"), normalize=False))

    assert me.shape == (1,)
    assert me[0] == pytest.approx(10.0 * H * W)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_normalization_divides_by_pixel_count(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames([0, 10], height=H, width=W))

    me, _, _, _ = _legacy(compute_motion_energy(Path("fake.mp4"), normalize=True))

    assert me[0] == pytest.approx(10.0)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_output_shapes(mock_info, mock_iter):
    H, W, N = 3, 4, 5
    rng = np.random.default_rng(0)
    frames = [(rng.integers(0, 255, (H, W), dtype=np.uint8), False) for _ in range(N)]
    mock_info.return_value = _info(width=W, height=H, n_frames=N)
    mock_iter.return_value = iter(frames)

    me, mask, avg_map, _ = _legacy(compute_motion_energy(Path("fake.mp4")))

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

    _, _, avg_map, _ = _legacy(compute_motion_energy(Path("fake.mp4"), normalize=False))

    np.testing.assert_allclose(avg_map, np.full((H, W), 10.0, dtype=np.float32))


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_metadata_fields(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=3, fps=60.0)
    mock_iter.return_value = iter(_frames([0, 5, 10], height=H, width=W))

    _, _, _, meta = _legacy(compute_motion_energy(Path("fake.mp4"), normalize=True))

    assert meta["n_frames_decoded"] == 3
    assert "n_me_frames" not in meta  # removed in 0.4.0 so old padding code fails loudly
    assert meta["fps"] == 60.0
    assert meta["codec_name"] == "h264"
    assert meta["normalized"] is True
    assert meta["n_rois"] == 0
    assert meta["width"] == W
    assert meta["height"] == H


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_frame_window_stored_in_metadata(mock_info, mock_iter):
    H, W = 3, 4
    mock_info.return_value = _info(width=W, height=H, n_frames=1000)
    mock_iter.return_value = iter(_frames([0, 5, 10], height=H, width=W))

    _, _, _, meta = _legacy(compute_motion_energy(Path("fake.mp4"), start_frame=100, end_frame=200))

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

    me, mask, _, meta = _legacy(compute_motion_energy(Path("fake.mp4"), normalize=False))

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

    _, mask, avg_map, _ = _legacy(compute_motion_energy(Path("fake.mp4"), normalize=False))

    # unmasked diffs: me[0]=10, me[3]=6  → map mean = 8
    np.testing.assert_array_equal(mask, [False, True, True, False])
    np.testing.assert_allclose(avg_map, np.full((H, W), 8.0, dtype=np.float32))


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_intra_only_codec_never_masks(mock_info, mock_iter):
    H, W = 2, 2
    # mjpeg: every frame is a keyframe, but nothing should be masked
    mock_info.return_value = _info(width=W, height=H, n_frames=3, codec_name="mjpeg")
    mock_iter.return_value = iter(_frames([0, 10, 20], keys=[True, True, True], height=H, width=W))

    _, mask, _, meta = _legacy(compute_motion_energy(Path("fake.mp4")))

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

    _, mask, _, _ = _legacy(compute_motion_energy(Path("fake.mp4"), mask_keyframes=False))

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


# --- ROIs ---------------------------------------------------------------------


def _roi_frames(n=4, h=6, w=8, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.integers(0, 255, (h, w), dtype=np.uint8), False) for _ in range(n)]


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_roi_columns_match_brute_force_masked_sums(mock_info, mock_iter):
    H, W = 6, 8
    frames = _roi_frames(h=H, w=W)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:H, :W]
    disk = (yy - 3) ** 2 + (xx - 4) ** 2 < 9
    masks = np.stack(
        [
            rng.integers(0, 2, (H, W)).astype(bool) | (yy + xx == 0),  # scattered, overlapping
            disk & ~((yy - 3) ** 2 + (xx - 4) ** 2 < 2),  # compound: a ring with a hole
            (yy == 5) & (xx == 7),  # a single pixel
        ],
        axis=-1,
    )
    rois = {"a": masks[..., 0], "b": masks[..., 1], "c": masks[..., 2]}
    mock_info.return_value = _info(width=W, height=H, n_frames=len(frames))
    mock_iter.return_value = iter(frames)

    res = compute_motion_energy(Path("fake.mp4"), rois=rois, normalize=True)

    assert res.roi_ids == ("full_frame", "a", "b", "c")
    assert res.me.shape == (len(frames), 4)  # frame-aligned: one row per frame
    assert np.isnan(res.me[0]).all()
    for i in range(1, len(frames)):
        diff = np.abs(frames[i][0].astype(int) - frames[i - 1][0].astype(int))
        assert res.me[i, 0] == pytest.approx(diff.mean())
        for k in range(3):
            assert res.me[i, k + 1] == pytest.approx(diff[masks[..., k]].mean())
    np.testing.assert_array_equal(res.pixel_counts, [H * W, *masks.sum(axis=(0, 1))])
    assert res.avg_map.shape == (H, W)
    assert res.metadata["n_rois"] == 3


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_box_roi_not_normalized_gives_raw_sum(mock_info, mock_iter):
    H, W = 10, 10
    mock_info.return_value = _info(width=W, height=H, n_frames=2)
    mock_iter.return_value = iter(_frames([0, 20], height=H, width=W))
    rois = box_roi((2, 2, 4, 3), (H, W))

    res = compute_motion_energy(Path("fake.mp4"), rois=rois, normalize=False, start_frame=7)

    np.testing.assert_allclose(res.me[1], [20.0 * H * W, 20.0 * 12])
    assert res.first_frame == 7


@patch("aind_motion_energy.compute.get_video_info")
def test_roi_frame_shape_mismatch_raises(mock_info):
    mock_info.return_value = _info(width=8, height=6)
    rois = box_roi((0, 0, 2, 2), (10, 10))
    with pytest.raises(ValueError, match="10x10"):
        compute_motion_energy(Path("fake.mp4"), rois=rois)


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_empty_decode_gives_empty_traces(mock_info, mock_iter):
    mock_info.return_value = _info(width=4, height=3)
    mock_iter.return_value = iter([])
    rois = box_roi((0, 0, 2, 2), (3, 4))

    res = compute_motion_energy(Path("fake.mp4"), rois=rois)

    assert res.me.shape == (0, 2)
    assert res.keyframe_mask.shape == (0,)


def test_clean_trace_2d_interpolates_every_column():
    me = np.array([[1, 10], [9, 9], [9, 9], [4, 40]], dtype=np.float32)
    mask = np.array([False, True, True, False])
    out = clean_trace(me, mask)
    np.testing.assert_allclose(out[:, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(out[:, 1], [10, 20, 30, 40])
    np.testing.assert_allclose(me[1], [9, 9])  # input untouched


def test_clean_trace_2d_nan_blanks_whole_rows():
    me = np.ones((3, 2), dtype=np.float32)
    out = clean_trace(me, np.array([False, True, False]), method="nan")
    assert np.isnan(out[1]).all() and not np.isnan(out[[0, 2]]).any()


@patch("aind_motion_energy.compute.iter_luma_frames")
@patch("aind_motion_energy.compute.get_video_info")
def test_buffers_grow_when_probe_undercounts_frames(mock_info, mock_iter):
    # ffprobe frame counts can be wrong; more decoded frames than probed must still fit.
    mock_info.return_value = _info(width=2, height=2, n_frames=2)
    fills = list(range(0, 70, 10))  # 7 frames -> 6 diffs of 10
    mock_iter.return_value = iter(
        _frames(fills, keys=[False, False, False, True] + [False] * 3, height=2, width=2)
    )
    rois = box_roi((0, 0, 1, 1), (2, 2))

    res = compute_motion_energy(Path("fake.mp4"), rois=rois, normalize=False)

    np.testing.assert_allclose(res.me[1:], [[40.0, 10.0]] * 6)
    np.testing.assert_array_equal(
        res.keyframe_mask, [False, False, False, True, True, False, False]
    )


@pytest.mark.parametrize(
    "start,end,match",
    [
        (-1, None, "start frame must be >= 0"),
        (10, 10, "must be after"),
        (10, 5, "must be after"),
        (None, 0, "must be after"),
    ],
)
def test_invalid_frame_range_raises_before_probing(start, end, match):
    with patch("aind_motion_energy.compute.get_video_info") as mock_info:
        with pytest.raises(ValueError, match=match):
            compute_motion_energy(Path("fake.mp4"), start_frame=start, end_frame=end)
    mock_info.assert_not_called()


def test_clean_trace_ignores_nan_first_frame():
    # Frame 0 is NaN; a keyframe diff right after it interpolates from later frames only.
    me = np.array([np.nan, 99.0, 3.0, 4.0], dtype=np.float32)
    mask = np.array([False, True, False, False])
    out = clean_trace(me, mask)
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1:], [3.0, 3.0, 4.0])


@patch("aind_motion_energy.compute.get_video_info")
def test_huge_frames_refused(mock_info):
    mock_info.return_value = _info(width=5000, height=4000)
    with pytest.raises(ValueError, match="too large"):
        compute_motion_energy(Path("fake.mp4"))
