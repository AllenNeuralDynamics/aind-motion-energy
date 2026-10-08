# aind-motion-energy

![support](https://img.shields.io/badge/support-supported-brightgreen)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![CI](https://github.com/AllenNeuralDynamics/aind-motion-energy/actions/workflows/test_lint_type.yml/badge.svg)](https://github.com/AllenNeuralDynamics/aind-motion-energy/actions/workflows/test_lint_type.yml)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
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
| `--extensions EXT [EXT ...]` | all video types | Only process these extensions when `--input` is a directory (e.g. `--extensions .mp4`). Case-insensitive; leading dot optional. |
| `--output` | `results/` | Output directory. |
| `--roi X Y W H` | none | One rectangular ROI in pixels, saved as ROI `roi`. Mutually exclusive with `--roi-masks`. |
| `--roi-masks PATH` | none | ROI annotator mask files: a directory of `*_<camera>_masks.npz` files (matched to each video by camera name), or one masks file when `--input` is a single video. See [ROIs](#rois). |
| `--no-normalize` | off | Skip per-pixel normalization (output raw summed differences instead of dividing each trace by its region's pixel count). |
| `--format {npz,csv,both}` | `npz` | `csv` or `both` also write a wide per-frame CSV; the NPZ is always written. |
| `--start-frame` | `0` | First frame to process (inclusive, 0-indexed). |
| `--end-frame` | end of video | Last frame to process (exclusive). |
| `--no-mask-keyframes` | off | Disable detection of H.264/HEVC keyframe-contaminated diffs. |
| `--clean-method {interpolate,nan}` | `interpolate` | How the cleaned trace handles keyframe-contaminated diffs. |
| `--summary-plots` | off | Also save two static QC PNGs per video (full-frame trace + ROI heatmap; average motion map with ROI outlines). Requires the `[viz]` extra. |
| `--visualize` | off | Also render an MP4 of the footage with a synced, scrolling motion-energy plot. Requires the `[viz]` extra. |
| `--viz-fps` | `60` | Playback fps of the `--visualize` video. |
| `--viz-rois ID [ID ...]` | `full_frame` + first 4 ROIs | Which traces scroll in the `--visualize` video. |
| `--viz-window-seconds` | `3.0` | Width of the scrolling plot window in seconds, for `--visualize`. |
| `--viz-stride` | `1` | Render every Nth frame in `--visualize` (`1` = lossless; higher values trade temporal resolution for render time / file size). |

## Outputs

For each processed video, files are written to `--output` under a shared name stem, `{stem}`:

| File | Written when | Contents |
|---|---|---|
| `{stem}_motion_energy.npz` | always | All numeric outputs (below). |
| `{stem}_me_metadata.json` | always | Video properties and processing parameters (fps, codec, dimensions, ROI ids and pixel counts, frame range, mask file path + SHA-256, `output_version`). |
| `{stem}_roi_masks.npz` | `--roi-masks` | Byte-for-byte copy of the input masks, for provenance. |
| `{stem}_motion_energy.csv` | `--format csv` or `both` | One row per frame: `frame_index, is_keyframe`, then `<id>, <id>_clean` for `full_frame` and each ROI. |
| `{stem}_motion_energy.png` | `--summary-plots` | QC figure: full-frame trace (raw under clean, keyframes ticked) and, with ROIs, an ROI × time heatmap with each ROI z-scored. |
| `{stem}_motion_energy_map.png` | `--summary-plots` | Average motion map with ROI outlines. |
| `{stem}_motion_energy.mp4` | `--visualize` | Rendered video with ROI outlines and synced, scrolling traces. |

`{stem}_motion_energy.npz` holds (`T` = frames, `K` = ROIs):

| Array | Shape, dtype | Contents |
|---|---|---|
| `me` | `(T, 1+K)` float32 | Raw motion energy. Column 0 is the whole frame (`full_frame`); columns follow `roi_ids`. |
| `me_clean` | `(T, 1+K)` float32 | `me` with keyframe-contaminated frames handled per `--clean-method`. |
| `keyframe_mask` | `(T,)` bool | Frames whose diff was keyframe-contaminated (shared by every column). |
| `frame_index` | `(T,)` int64 | Video frame number of each row. |
| `roi_ids` | `(1+K,)` str | Column names, starting with `full_frame`. |
| `pixel_counts` | `(1+K,)` int64 | Pixels in each region (the normalization divisor). |
| `roi_bboxes` | `(1+K, 4)` int32 | Each region's `(x, y, w, h)` bounding box. |
| `avg_map` | `(H, W)` float32 | Whole-frame mean absolute difference per pixel, excluding keyframe-contaminated diffs. |

```python
import numpy as np

with np.load("results/BottomCamera_motion_energy.npz") as f:
    me = f["me_clean"]
    ids = f["roi_ids"].tolist()
snout = me[:, ids.index("BottomCamera_Snout_box")]
```

Runs without ROIs use the same layout with `K = 0`. The file is a standard uncompressed NPZ (a ZIP of `.npy` files), readable without `allow_pickle`.

### Frame alignment

Saved traces have one value per decoded frame. Each frame-to-frame diff `|f[i] - f[i-1]|` is assigned to the later frame `i`, and the first frame (which has no predecessor) is `0` (`False` in the keyframe mask). Index `i` of every output therefore lines up with video frame `start_frame + i` (also stored as `frame_index`), matching frame timestamps and other per-frame data. The library function `compute_motion_energy` still returns the raw per-diff arrays (length `N - 1`); `align_to_frames` performs the padding.

## ROIs

ROIs of any shape are passed in as **binary masks** — rectangles, polygons and compound shapes alike. Draw them in the ROI annotator, which exports one `{session}_{camera}_masks.npz` per camera containing `masks` (`(H, W, K)`, 0/1) and `roi_ids` (`(K,)`). Masks must match the video's frame size.

```bash
uv run aind-motion-energy --input /data/session --roi-masks /data/session/rois --summary-plots
```

- **Matching.** With a directory, each video uses the file named `<camera>_masks.npz` or `<anything>_<camera>_masks.npz`, where `<camera>` is the video's `{stem}` (see below). If several cameras fit one file name, the longest camera name wins. A camera with no file gets the whole-frame trace only (with a warning); a camera matching several files is an error. In older flat layouts the annotator's camera name can differ from `{stem}`; rename the mask file to `*_{stem}_masks.npz`, or pass the file directly for a single video.
- **Checked up front.** Every video's masks are loaded and checked against its frame size before any video is processed.
- **Speed.** Each frame is decoded and differenced once; all ROIs are summed from that one diff in a single vectorized call, so cost grows with the total number of masked pixels, not the number of ROIs. On the 720×540 sample clip, 100 ROIs of ~3,000 px each run at ~55% of whole-frame-only throughput.
- **Size guard.** A warning is printed when masks together cover more than 2× the frame area (and above 256 ROIs); more than 20× is refused as a likely bad export.

From Python:

```python
from aind_motion_energy import clean_trace, compute_motion_energy, load_roi_masks

rois = load_roi_masks("rois/session_BottomCamera_masks.npz")
result = compute_motion_energy("BottomCamera/video.mp4", rois=rois)
me_clean = clean_trace(result.me, result.keyframe_mask)  # (N-1, 1+K), per diff
```

### Camera-keying heuristic

`{stem}` is not always the input file's stem. AIND's video layout nests each camera as `<CameraName>/video.mp4`, so every camera in a session shares the literal stem `"video"` and would silently overwrite the others' outputs. To avoid that collision, the CLI keys outputs on the camera identity instead: if the file stem is exactly `"video"`, `{stem}` is the parent directory name (the camera name); otherwise `{stem}` is the file's own stem, which already carries the camera identity in older flat layouts (e.g. `bottom_camera.avi`).

```python
stem = video.parent.name if video.stem == "video" else video.stem
```

Older flat layouts can store the same camera twice (e.g. `bottom_camera.avi` and `bottom_camera.mp4`), which would map to the same `{stem}`. The CLI refuses to start in that case and lists the colliding files; pick one copy per camera with `--extensions`.

The [capsule](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) depends on this convention when matching outputs back to input cameras — changing it is a breaking change for that repo.

## Examples

Runnable examples live in [`examples/`](examples/).

## Related repos

- [`aind-motion-energy-capsule`](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) — Code Ocean capsule; a thin shell that forwards its arguments to this library's CLI.
- [`aind-motion-energy-batch`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch) — Code Ocean launcher; fans out one capsule run per session.

## Adding dependencies

```bash
uv add <package>
```

The capsule installs this library directly from a pinned git commit/tag (`pip install "aind-motion-energy[viz] @ git+https://github.com/AllenNeuralDynamics/aind-motion-energy.git@<ref>"`), so `pyproject.toml` and `uv.lock` are the only sources of truth for dependencies — there's no `environment/requirements.txt` to regenerate.
