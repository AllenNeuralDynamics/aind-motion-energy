# aind-motion-energy

![support](https://img.shields.io/badge/support-supported-brightgreen)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![CI](https://github.com/AllenNeuralDynamics/aind-motion-energy/actions/workflows/test_lint_type.yml/badge.svg)](https://github.com/AllenNeuralDynamics/aind-motion-energy/actions/workflows/test_lint_type.yml)
![Coverage](https://img.shields.io/badge/coverage-68%25-yellow)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)

Motion energy computation from behavior videos for neuroscience pipelines. Targets Code Ocean deployment.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv sync --dev
```

The `[viz]` extra (`matplotlib`) is needed for `--summary-plots` and `--visualize`; it's included by `--dev`. To install just the library, with or without `[viz]`:

```bash
pip install aind-motion-energy
pip install "aind-motion-energy[viz]"
```

## CLI usage

```bash
uv run aind-motion-energy --input data/my_video.mp4 --output results/ --summary-plots
```

`--input` can be a single video file or a directory, searched recursively for `.mp4`, `.avi`, `.mkv`, `.mov`, `.mj2`, `.tif`, and `.tiff` files.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Video file or directory of videos. |
| `--output` | `results/` | Output directory. |
| `--roi X Y W H` | full frame | Region of interest in pixels. |
| `--no-normalize` | off | Skip per-pixel normalization (output raw summed differences instead of dividing by pixel count). |
| `--format {npy,csv,both}` | `npy` | Output format for the motion-energy trace. |
| `--start-frame` | `0` | First frame to process (inclusive, 0-indexed). |
| `--end-frame` | end of video | Last frame to process (exclusive). |
| `--no-mask-keyframes` | off | Disable detection of H.264/HEVC keyframe-contaminated diffs. |
| `--clean-method {interpolate,nan}` | `interpolate` | How the cleaned trace handles keyframe-contaminated diffs. |
| `--summary-plots` | off | Also save two static PNG summaries per video (motion-energy timeseries + average motion map). Requires the `[viz]` extra. |
| `--visualize` | off | Also render an MP4 of the footage with a synced, scrolling motion-energy plot. Requires the `[viz]` extra. |
| `--viz-fps` | `60` | Playback fps of the `--visualize` video. |
| `--viz-window-seconds` | `3.0` | Width of the scrolling plot window in seconds, for `--visualize`. |
| `--viz-stride` | `1` | Render every Nth frame in `--visualize` (`1` = lossless; higher values trade temporal resolution for render time / file size). |

## Outputs

For each processed video, files are written to `--output` under a shared name stem, `{stem}`:

| File | Written when | Contents |
|---|---|---|
| `{stem}_motion_energy.npy` | always | Raw motion-energy trace (`float32`, one value per frame-to-frame diff). |
| `{stem}_motion_energy_clean.npy` | always | Cleaned trace with keyframe-contaminated diffs handled per `--clean-method`. |
| `{stem}_keyframe_mask.npy` | always | Bool mask flagging which diffs were keyframe-contaminated. |
| `{stem}_motion_energy_map.npy` | always | Per-pixel average absolute frame difference, shape `(H, W)`. |
| `{stem}_me_metadata.json` | always | Video properties and processing parameters (fps, codec, dimensions, ROI, frame range, etc.). |
| `{stem}_motion_energy.csv` | `--format csv` or `both` | Per-frame `frame_index, motion_energy, motion_energy_clean, is_keyframe`. |
| `{stem}_motion_energy.png` | `--summary-plots` | Motion-energy timeseries plot. |
| `{stem}_motion_energy_map.png` | `--summary-plots` | Heatmap of `{stem}_motion_energy_map.npy`. |
| `{stem}_motion_energy.mp4` | `--visualize` | Rendered video with a synced, scrolling motion-energy plot. |

### Camera-keying heuristic

`{stem}` is not always the input file's stem. AIND's video layout nests each camera as `<CameraName>/video.mp4`, so every camera in a session shares the literal stem `"video"` and would silently overwrite the others' outputs. To avoid that collision, the CLI keys outputs on the camera identity instead: if the file stem is exactly `"video"`, `{stem}` is the parent directory name (the camera name); otherwise `{stem}` is the file's own stem, which already carries the camera identity in older flat layouts (e.g. `bottom_camera.avi`).

```python
stem = video.parent.name if video.stem == "video" else video.stem
```

The [capsule](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) depends on this convention when matching outputs back to input cameras — changing it is a breaking change for that repo.

## Related repos

- [`aind-motion-energy-capsule`](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) — Code Ocean capsule; a thin shell that forwards its arguments to this library's CLI.
- [`aind-motion-energy-batch`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch) — Code Ocean launcher; fans out one capsule run per session.

## Adding dependencies

```bash
uv add <package>
```

The capsule installs this library directly from a pinned git commit/tag (`pip install "aind-motion-energy[viz] @ git+https://github.com/AllenNeuralDynamics/aind-motion-energy.git@<ref>"`), so `pyproject.toml` and `uv.lock` are the only sources of truth for dependencies — there's no `environment/requirements.txt` to regenerate.
