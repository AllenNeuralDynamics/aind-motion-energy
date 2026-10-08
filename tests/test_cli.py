import csv
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest

from aind_motion_energy.cli import main
from aind_motion_energy.compute import MotionEnergyResult


def _fake_compute_result(n=4, fps=30.0, roi_ids=("full_frame",), me=None, keyframe_mask=None):
    k = len(roi_ids)
    if me is None:
        me = np.arange(n * k, dtype=np.float32).reshape(n, k)
    if keyframe_mask is None:
        keyframe_mask = np.zeros(len(me), dtype=bool)
    return MotionEnergyResult(
        me=me,
        keyframe_mask=keyframe_mask,
        roi_ids=tuple(roi_ids),
        pixel_counts=np.full(k, 4, dtype=np.int64),
        roi_bboxes=np.zeros((k, 4), dtype=np.int32),
        avg_map=np.zeros((2, 2), dtype=np.float32),
        metadata={"fps": fps, "n_keyframes_masked": int(keyframe_mask.sum())},
    )


def _info(width=8, height=6):
    return {"width": width, "height": height, "n_frames": 5, "fps": 30.0}


# --- argument parsing -------------------------------------------------------


def test_missing_required_input_raises_systemexit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    with pytest.raises(SystemExit):
        main()


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_default_output_directory_created_when_omitted(
    mock_compute, mock_clean, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.touch()
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(video)])
    main()

    assert (tmp_path / "results").is_dir()


@patch("aind_motion_energy.cli.get_video_info", return_value=_info())
@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_optional_flags_pass_through_to_compute(
    mock_compute, mock_clean, mock_info, monkeypatch, tmp_path
):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    argv = [
        "prog",
        "--input",
        str(video),
        "--output",
        str(out),
        "--roi",
        "1",
        "2",
        "3",
        "4",
        "--no-normalize",
        "--start-frame",
        "5",
        "--end-frame",
        "50",
        "--no-mask-keyframes",
        "--clean-method",
        "nan",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    _, kwargs = mock_compute.call_args
    assert kwargs["rois"].ids == ("roi",)
    np.testing.assert_array_equal(kwargs["rois"].bboxes[0], [1, 2, 3, 4])
    assert kwargs["normalize"] is False
    assert kwargs["start_frame"] == 5
    assert kwargs["end_frame"] == 50
    assert kwargs["mask_keyframes"] is False

    _, clean_kwargs = mock_clean.call_args
    assert clean_kwargs["method"] == "nan"


# --- video discovery ---------------------------------------------------------


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_single_file_input_is_discovered_directly(mock_compute, mock_clean, monkeypatch, tmp_path):
    video = tmp_path / "clip.mkv"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(video), "--output", str(out)])
    main()

    mock_compute.assert_called_once()
    assert mock_compute.call_args.args[0] == video


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_directory_discovers_video_extensions_recursively(
    mock_compute, mock_clean, monkeypatch, tmp_path
):
    input_dir = tmp_path / "in"
    (input_dir / "sub").mkdir(parents=True)
    a = input_dir / "a.mp4"
    a.touch()
    b = input_dir / "sub" / "b.AVI"
    b.touch()
    (input_dir / "readme.txt").touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(input_dir), "--output", str(out)])
    main()

    called_videos = [c.args[0] for c in mock_compute.call_args_list]
    assert called_videos == sorted([a, b])


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_no_videos_found_prints_message_and_returns(mock_compute, monkeypatch, tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").touch()
    out = tmp_path / "out"

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(empty_dir), "--output", str(out)])
    main()

    mock_compute.assert_not_called()
    assert "No videos found" in capsys.readouterr().out


# --- camera-keying heuristic --------------------------------------------------


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_camera_keying_uses_parent_name_when_stem_is_video(
    mock_compute, mock_clean, monkeypatch, tmp_path
):
    cam_dir = tmp_path / "in" / "TopCam"
    cam_dir.mkdir(parents=True)
    (cam_dir / "video.mp4").touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(
        sys, "argv", ["prog", "--input", str(tmp_path / "in"), "--output", str(out)]
    )
    main()

    assert (out / "TopCam_motion_energy.npz").exists()


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_camera_keying_uses_own_stem_when_not_video(
    mock_compute, mock_clean, monkeypatch, tmp_path
):
    video = tmp_path / "bottom_camera.avi"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(video), "--output", str(out)])
    main()

    assert (out / "bottom_camera_motion_energy.npz").exists()


# --- --extensions filter and output-key collisions ----------------------------


def _flat_session(tmp_path):
    """Old flat AIND layout: each camera saved as both .avi and .mp4."""
    vids = tmp_path / "in" / "behavior-videos"
    vids.mkdir(parents=True)
    for name in ("bottom_camera.avi", "bottom_camera.mp4", "side_camera_right.avi"):
        (vids / name).touch()
    (vids / "side_camera_right.MP4").touch()
    return tmp_path / "in"


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_duplicate_camera_keys_error_before_processing(mock_compute, monkeypatch, tmp_path, capsys):
    input_dir = _flat_session(tmp_path)

    monkeypatch.setattr(
        sys, "argv", ["prog", "--input", str(input_dir), "--output", str(tmp_path / "out")]
    )
    with pytest.raises(SystemExit):
        main()

    mock_compute.assert_not_called()
    err = capsys.readouterr().err
    assert "bottom_camera" in err and "side_camera_right" in err
    assert "--extensions" in err


@pytest.mark.parametrize("ext", [".mp4", "mp4", ".MP4"])
@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_extensions_filter_selects_one_copy_per_camera(
    mock_compute, mock_clean, monkeypatch, tmp_path, ext
):
    input_dir = _flat_session(tmp_path)
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--input", str(input_dir), "--output", str(out), "--extensions", ext],
    )
    main()

    called = sorted(c.args[0].name for c in mock_compute.call_args_list)
    assert called == ["bottom_camera.mp4", "side_camera_right.MP4"]
    assert (out / "bottom_camera_motion_energy.npz").exists()
    assert (out / "side_camera_right_motion_energy.npz").exists()


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_same_filename_in_distinct_camera_folders_does_not_collide(
    mock_compute, mock_clean, monkeypatch, tmp_path
):
    for cam in ("BottomCamera", "SideCameraRight"):
        (tmp_path / "in" / cam).mkdir(parents=True)
        (tmp_path / "in" / cam / "video.mp4").touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    monkeypatch.setattr(
        sys, "argv", ["prog", "--input", str(tmp_path / "in"), "--output", str(out)]
    )
    main()

    assert mock_compute.call_count == 2
    assert (out / "BottomCamera_motion_energy.npz").exists()
    assert (out / "SideCameraRight_motion_energy.npz").exists()


# --- --format branches --------------------------------------------------------


@pytest.mark.parametrize("fmt,expect_csv", [("npz", False), ("csv", True), ("both", True)])
@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_format_branches(mock_compute, mock_clean, monkeypatch, tmp_path, fmt, expect_csv):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)

    argv = ["prog", "--input", str(video), "--output", str(out), "--format", fmt]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    assert (out / "clip_motion_energy.npz").exists()
    assert (out / "clip_motion_energy.csv").exists() == expect_csv


# --- optional-output branches -------------------------------------------------


@patch("aind_motion_energy.viz.save_summary_plots")
@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_summary_plots_flag_invokes_save_summary_plots(
    mock_compute, mock_clean, mock_save, monkeypatch, tmp_path, capsys
):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    out.mkdir()
    mock_compute.return_value = _fake_compute_result(fps=30.0)
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)
    mock_save.return_value = (out / "clip_motion_energy.png", out / "clip_motion_energy_map.png")

    argv = ["prog", "--input", str(video), "--output", str(out), "--summary-plots"]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    mock_save.assert_called_once()
    args, kwargs = mock_save.call_args
    assert args[0] == out
    assert args[1] == "clip"
    assert kwargs["fps"] == 30.0
    assert "saved" in capsys.readouterr().out


@patch("aind_motion_energy.viz.render_motion_energy_video")
@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_visualize_flag_invokes_render_motion_energy_video(
    mock_compute, mock_clean, mock_render, monkeypatch, tmp_path, capsys
):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    out.mkdir()
    mock_compute.return_value = _fake_compute_result(fps=30.0)
    mock_clean.return_value = np.zeros((4, 1), dtype=np.float32)
    mock_render.return_value = out / "clip_motion_energy.mp4"

    argv = ["prog", "--input", str(video), "--output", str(out), "--visualize"]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    mock_render.assert_called_once()
    _, kwargs = mock_render.call_args
    assert kwargs["fps_source"] == 30.0
    assert kwargs["output_path"] == out / "clip_motion_energy.mp4"
    assert "rendered" in capsys.readouterr().out


# --- frame alignment ----------------------------------------------------------


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_outputs_are_frame_aligned(mock_compute, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    me = np.array([[5.0], [6.0], [7.0]], dtype=np.float32)  # 4 frames -> 3 diffs
    mask = np.array([False, True, False])
    mock_compute.return_value = _fake_compute_result(me=me, keyframe_mask=mask)

    argv = ["prog", "--input", str(video), "--output", str(out), "--format", "both"]
    monkeypatch.setattr(sys, "argv", argv + ["--start-frame", "10"])
    main()

    with np.load(out / "clip_motion_energy.npz") as f:
        np.testing.assert_array_equal(f["me"][:, 0], [np.nan, 5.0, 6.0, 7.0])
        assert np.isnan(f["me_clean"][0]).all()
        np.testing.assert_array_equal(f["keyframe_mask"], [False, False, True, False])
        np.testing.assert_array_equal(f["frame_index"], [10, 11, 12, 13])
    with open(out / "clip_motion_energy.csv") as f:
        rows = list(csv.reader(f))
    assert [r[0] for r in rows[1:]] == ["10", "11", "12", "13"]
    meta = json.loads((out / "clip_me_metadata.json").read_text())
    assert meta["n_output_frames"] == 4
    assert meta["first_frame_fill"] is None
    assert rows[1][2] == "nan"


# --- NPZ output and metadata ---------------------------------------------------


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_npz_contents_and_metadata_without_rois(mock_compute, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(video), "--output", str(out)])
    main()

    with np.load(out / "clip_motion_energy.npz") as f:
        assert set(f.files) == {
            "me",
            "me_clean",
            "keyframe_mask",
            "frame_index",
            "roi_ids",
            "pixel_counts",
            "roi_bboxes",
            "avg_map",
        }
        assert f["me"].shape == (5, 1)
        assert f["roi_ids"].tolist() == ["full_frame"]
    assert mock_compute.call_args.kwargs["rois"] is None
    meta = json.loads((out / "clip_me_metadata.json").read_text())
    assert meta["output_version"] == 2
    assert meta["roi_masks_path"] is None and meta["roi_box"] is None
    assert not (out / "clip_roi_masks.npz").exists()
    assert not list(out.glob(".*.tmp"))


def test_save_npz_atomic_removes_temp_file_on_failure(tmp_path, monkeypatch):
    from aind_motion_energy import cli

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli.np, "savez", boom)
    with pytest.raises(OSError):
        cli.save_npz_atomic(tmp_path / "x.npz", a=np.zeros(1))
    assert list(tmp_path.iterdir()) == []


# --- ROI masks -------------------------------------------------------------------


def _write_masks(path, ids=("Snout", "Pupil"), h=6, w=8):
    masks = np.zeros((h, w, len(ids)), np.uint8)
    for k in range(len(ids)):
        masks[k, : k + 2, k] = 1
    np.savez_compressed(path, masks=masks, roi_ids=np.array(list(ids)))
    return path


def _nested_session(tmp_path, cams=("BottomCamera", "SideCamera")):
    for cam in cams:
        (tmp_path / "in" / cam).mkdir(parents=True)
        (tmp_path / "in" / cam / "video.mp4").touch()
    masks_dir = tmp_path / "rois"
    masks_dir.mkdir()
    return tmp_path / "in", masks_dir


@patch("aind_motion_energy.cli.get_video_info", return_value=_info())
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_roi_masks_dir_matched_by_camera_and_recorded(
    mock_compute, mock_info, monkeypatch, tmp_path
):
    input_dir, masks_dir = _nested_session(tmp_path)
    masks_file = _write_masks(masks_dir / "sess_2026-01-01_BottomCamera_masks.npz")
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result(roi_ids=("full_frame", "Snout", "Pupil"))

    argv = ["prog", "--input", str(input_dir), "--output", str(out), "--roi-masks", str(masks_dir)]
    monkeypatch.setattr(sys, "argv", argv + ["--format", "csv"])
    main()

    by_video = {c.args[0].parent.name: c.kwargs["rois"] for c in mock_compute.call_args_list}
    assert by_video["BottomCamera"].ids == ("Snout", "Pupil")
    assert by_video["SideCamera"] is None
    assert (out / "BottomCamera_roi_masks.npz").read_bytes() == masks_file.read_bytes()
    meta = json.loads((out / "BottomCamera_me_metadata.json").read_text())
    assert meta["roi_masks_path"] == str(masks_file)
    assert len(meta["roi_masks_sha256"]) == 64
    with open(out / "BottomCamera_motion_energy.csv") as f:
        header = next(csv.reader(f))
    assert header == [
        "frame_index",
        "is_keyframe",
        "full_frame",
        "full_frame_clean",
        "Snout",
        "Snout_clean",
        "Pupil",
        "Pupil_clean",
    ]


@patch("aind_motion_energy.cli.get_video_info", return_value=_info())
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_missing_mask_file_warns_and_runs_full_frame(
    mock_compute, mock_info, monkeypatch, tmp_path, capsys
):
    input_dir, masks_dir = _nested_session(tmp_path, cams=("SideCamera",))
    mock_compute.return_value = _fake_compute_result()

    argv = ["prog", "--input", str(input_dir), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--roi-masks", str(masks_dir)])
    main()

    assert mock_compute.call_args.kwargs["rois"] is None
    assert "no '*_SideCamera_masks.npz'" in capsys.readouterr().out


@patch("aind_motion_energy.cli.get_video_info", return_value=_info())
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_ambiguous_mask_files_error_before_processing(
    mock_compute, mock_info, monkeypatch, tmp_path, capsys
):
    input_dir, masks_dir = _nested_session(tmp_path, cams=("SideCamera",))
    _write_masks(masks_dir / "a_SideCamera_masks.npz")
    _write_masks(masks_dir / "b_SideCamera_masks.npz")

    argv = ["prog", "--input", str(input_dir), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--roi-masks", str(masks_dir)])
    with pytest.raises(SystemExit):
        main()

    mock_compute.assert_not_called()
    assert "several mask files" in capsys.readouterr().err


@patch("aind_motion_energy.cli.get_video_info", return_value=_info(width=10))
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_mask_shape_mismatch_errors_before_processing(
    mock_compute, mock_info, monkeypatch, tmp_path, capsys
):
    input_dir, masks_dir = _nested_session(tmp_path, cams=("SideCamera",))
    _write_masks(masks_dir / "s_SideCamera_masks.npz")  # 8x6, video is 10x6

    argv = ["prog", "--input", str(input_dir), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--roi-masks", str(masks_dir)])
    with pytest.raises(SystemExit):
        main()

    mock_compute.assert_not_called()
    assert "8x6 but the video is 10x6" in capsys.readouterr().err


@patch("aind_motion_energy.cli.get_video_info", return_value=_info())
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_single_masks_file_for_single_video(mock_compute, mock_info, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    masks_file = _write_masks(tmp_path / "anything_masks.npz")
    mock_compute.return_value = _fake_compute_result(roi_ids=("full_frame", "Snout", "Pupil"))

    argv = ["prog", "--input", str(video), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--roi-masks", str(masks_file)])
    main()

    assert mock_compute.call_args.kwargs["rois"].ids == ("Snout", "Pupil")


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_single_masks_file_rejected_for_several_videos(mock_compute, monkeypatch, tmp_path, capsys):
    input_dir, masks_dir = _nested_session(tmp_path)
    masks_file = _write_masks(masks_dir / "x_masks.npz")

    argv = ["prog", "--input", str(input_dir), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--roi-masks", str(masks_file)])
    with pytest.raises(SystemExit):
        main()

    mock_compute.assert_not_called()
    assert "single file" in capsys.readouterr().err


def test_roi_and_roi_masks_are_mutually_exclusive(monkeypatch, tmp_path):
    argv = ["prog", "--input", str(tmp_path), "--roi", "0", "0", "1", "1", "--roi-masks", "x"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


def test_assign_mask_files_prefers_longest_camera_key(tmp_path):
    from aind_motion_energy.cli import assign_mask_files

    for name in ("s_camera_masks.npz", "s_side_camera_masks.npz", "camera_masks.npz", "x.npz"):
        (tmp_path / name).touch()

    got = assign_mask_files(tmp_path, ["camera", "side_camera", "unused"])

    assert got == {
        "camera": [tmp_path / "camera_masks.npz", tmp_path / "s_camera_masks.npz"],
        "side_camera": [tmp_path / "s_side_camera_masks.npz"],
    }


# --- --viz-rois ------------------------------------------------------------------


@patch("aind_motion_energy.viz.render_motion_energy_video")
@patch("aind_motion_energy.cli.get_video_info", return_value=_info())
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_viz_rois_filtered_per_video(mock_compute, mock_info, mock_render, monkeypatch, tmp_path):
    input_dir, masks_dir = _nested_session(tmp_path)
    _write_masks(masks_dir / "s_BottomCamera_masks.npz")
    out = tmp_path / "out"
    out.mkdir()

    def fake_compute(video, rois=None, **kwargs):
        ids = ("full_frame", *rois.ids) if rois else ("full_frame",)
        return _fake_compute_result(roi_ids=ids)

    mock_compute.side_effect = fake_compute
    mock_render.return_value = out / "x.mp4"

    argv = ["prog", "--input", str(input_dir), "--output", str(out), "--visualize"]
    monkeypatch.setattr(sys, "argv", argv + ["--roi-masks", str(masks_dir), "--viz-rois", "Pupil"])
    main()

    selections = [c.kwargs["viz_rois"] for c in mock_render.call_args_list]
    assert selections == [["Pupil"], None]  # BottomCamera has Pupil; SideCamera doesn't


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_unknown_viz_rois_error_before_processing(mock_compute, monkeypatch, tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    video.touch()

    argv = ["prog", "--input", str(video), "--output", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv + ["--visualize", "--viz-rois", "Snoot"])
    with pytest.raises(SystemExit):
        main()

    mock_compute.assert_not_called()
    assert "Snoot" in capsys.readouterr().err


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_summary_line_ignores_nan_first_frame(mock_compute, monkeypatch, tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    video.touch()
    me = np.array([[2.0], [4.0]], dtype=np.float32)
    mock_compute.return_value = _fake_compute_result(me=me)

    monkeypatch.setattr(
        sys, "argv", ["prog", "--input", str(video), "--output", str(tmp_path / "o")]
    )
    main()

    assert "max=4.0000 mean=3.0000" in capsys.readouterr().out


@patch("aind_motion_energy.cli.compute_motion_energy")
def test_summary_line_with_no_diffs(mock_compute, monkeypatch, tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    video.touch()
    mock_compute.return_value = _fake_compute_result(me=np.zeros((0, 1), np.float32))

    monkeypatch.setattr(
        sys, "argv", ["prog", "--input", str(video), "--output", str(tmp_path / "o")]
    )
    main()

    out = capsys.readouterr().out
    assert "1 frames" in out and "max=nan mean=nan" in out
