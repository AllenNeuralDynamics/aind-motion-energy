import sys
from unittest.mock import patch

import numpy as np
import pytest

from aind_motion_energy.cli import main


def _fake_compute_result(n=4, fps=30.0):
    me = np.arange(n, dtype=np.float32)
    keyframe_mask = np.zeros(n, dtype=bool)
    avg_map = np.zeros((2, 2), dtype=np.float32)
    meta = {"fps": fps, "n_keyframes_masked": 0}
    return me, keyframe_mask, avg_map, meta


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
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(video)])
    main()

    assert (tmp_path / "results").is_dir()


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_optional_flags_pass_through_to_compute(mock_compute, mock_clean, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

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
    assert kwargs["roi"] == (1, 2, 3, 4)
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
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

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
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

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
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(
        sys, "argv", ["prog", "--input", str(tmp_path / "in"), "--output", str(out)]
    )
    main()

    assert (out / "TopCam_motion_energy.npy").exists()


@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_camera_keying_uses_own_stem_when_not_video(
    mock_compute, mock_clean, monkeypatch, tmp_path
):
    video = tmp_path / "bottom_camera.avi"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(video), "--output", str(out)])
    main()

    assert (out / "bottom_camera_motion_energy.npy").exists()


# --- --format branches --------------------------------------------------------


@pytest.mark.parametrize("fmt,expect_csv", [("npy", False), ("csv", True), ("both", True)])
@patch("aind_motion_energy.cli.clean_trace")
@patch("aind_motion_energy.cli.compute_motion_energy")
def test_format_branches(mock_compute, mock_clean, monkeypatch, tmp_path, fmt, expect_csv):
    video = tmp_path / "clip.mp4"
    video.touch()
    out = tmp_path / "out"
    mock_compute.return_value = _fake_compute_result()
    mock_clean.return_value = np.zeros(4, dtype=np.float32)

    argv = ["prog", "--input", str(video), "--output", str(out), "--format", fmt]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    assert (out / "clip_motion_energy.npy").exists()
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
    mock_clean.return_value = np.zeros(4, dtype=np.float32)
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
    mock_clean.return_value = np.zeros(4, dtype=np.float32)
    mock_render.return_value = out / "clip_motion_energy.mp4"

    argv = ["prog", "--input", str(video), "--output", str(out), "--visualize"]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    mock_render.assert_called_once()
    _, kwargs = mock_render.call_args
    assert kwargs["fps_source"] == 30.0
    assert kwargs["output_path"] == out / "clip_motion_energy.mp4"
    assert "rendered" in capsys.readouterr().out
