# Multi-ROI motion energy (binary masks as the common pathway)

> **Status:** implemented on `feat/multi-roi-masks`. Deviations from the original
> plan, each found during implementation, are marked **[as built]** below.

## Context
Today the library computes one motion-energy trace for one rectangle (`--roi X Y W H`) or for the full frame. Users want:
- many ROIs (~100);
- arbitrary shapes;
- a single decode pass in which every ROI is reduced from the same frame diff.

ROIs are drawn in the separate annotator app (`generate_roi_html.py` / `export_rois.py`, not in this repo), which already exports one `{session}_{camera}_masks.npz` per camera containing `masks` (H×W×K uint8) and `roi_ids` (K,). **That NPZ is the input contract.** All rasterization stays in the ROI manager, and this library only consumes binary masks. Rectangles are just masks, with a fast path.

Agreed decisions:
- **Output format:** one uncompressed NPZ per video, plus the JSON sidecar.
- **Unified layout:** runs without masks use the same layout, with K=1 and `full_frame`.
- **Average map:** full frame.
- **Full-frame column:** always included.
- **ROI outlines:** drawn on the map PNG and in the MP4.
- **CSV:** wide.
- **Release:** a breaking 0.4.0.

**Prerequisite:** land the deferred frame-alignment change first as its own PR (outputs length N, first frame = 0; see the memory `todo-frame-align-me-outputs`). Everything below assumes length-N outputs. **[as built]** Frame 0 is NaN rather than 0 (`False` in the keyframe mask). Its motion energy is undefined, and a zero would look like a real "no motion" frame, bias means, and create an outlier at t = 0 under z-scoring. Plots and the CLI summary are NaN-aware.

## Performance model (one decode, shared diff)
Each frame is processed as follows:
1. Decode the full luma frame with the existing `iter_luma_frames`, with no crop.
2. Compute `diff = |f - prev|` as int16 over the full frame. This happens once, and it is needed anyway for the full-frame column and the map.
3. Reduce the full-frame column with `diff.sum()`.
4. Reduce all ROIs in **one vectorized call**: `np.add.reduceat(diff.ravel()[flat_idx], offsets, dtype=uint32)`. Here `flat_idx` is the precomputed concatenation of every ROI's mask pixel indices, and `offsets` holds each ROI's start position. The cost is the total mask pixel count, there is no Python loop over K, and no new dependency is needed. **[as built]** The kernel was first written with `take` and int32 indices and an int64 accumulator, which measured 2.6 ms/frame for 100 ROIs. Switching to `intp` fancy indexing and a uint32 accumulator (uint64 for masks over ~16.8M px) brought that to 0.64 ms. Numbers are in `docs/design-notes.md`.
5. Accumulate the map into a **uint32** accumulator instead of float64. The result is still exact (no overflow below ~16M frames) and uses half the memory bandwidth. This addresses the "~45% runtime" map cost in `docs/optimization-backlog.md`. The diff is computed regardless; the cost is in the accumulation.

Empty masks are rejected at load time, because `reduceat` misbehaves on zero-length segments. Optimizations to benchmark later and record in the backlog:
- a slice-sum fast path for exact rectangles;
- a summed-area table for many rectangles;
- a `scipy.sparse` matmul.

## Changes

### New `src/aind_motion_energy/rois.py`
- `RoiSet` (frozen dataclass) with these fields:
  - `ids: tuple[str, ...]`
  - `frame_shape: (H, W)`
  - `bboxes: (K,4) int32`
  - `pixel_counts: (K,) int64`
  - `flat_idx: int32[]`
  - `offsets: int64[]`

  It is built from masks and keeps nothing dense, so 100 full-res masks don't stay resident.
- `RoiSet.from_masks(masks, roi_ids)` validates the input:
  - 3-D and binary;
  - K ≥ 1;
  - unique, non-empty ids;
  - no empty masks;
  - `"full_frame"` is a reserved id.
- `RoiSet.from_box(x, y, w, h, frame_shape)` keeps `--roi` working as one rectangular mask with id `"roi"`.
- `load_roi_masks(path) -> RoiSet` reads the annotator NPZ with `np.load(..., allow_pickle=False)` as a context manager.
- `RoiSet.check_frame_shape(H, W)` raises a clear error if the masks don't match the video.
- `RoiSet.reduce(diff) -> int64[K]` is the kernel from step 4.
- **Size guard on total mask area, not on K.** Both the per-frame cost and the `flat_idx` memory scale with `sum(pixel_counts)`, not with K.
  - **Warning** when the total exceeds 2× the frame area. **[as built]** Lowered from 4× after the benchmark: at 2× the frame area throughput is about 30% of whole-frame-only, and at 4× it's about 16%. The message prints the total, the multiple of the frame area, and the estimated `flat_idx` memory.
  - **Error** above 20× the frame area, which catches garbage input such as a mask with every pixel set in every channel.
  - **Warning** when K > 256, about output size and plot readability.
  - These thresholds are module constants and will be tuned after the benchmark.

### `compute.py`
- `compute_motion_energy(video_path, rois: RoiSet | None = None, normalize=True, start_frame, end_frame, mask_keyframes=True) -> MotionEnergyResult` (dataclass). The `roi=` tuple param is removed (pre-1.0 breaking change); callers use `RoiSet.from_box`.
- `MotionEnergyResult` has these fields:
  - `me`: (T, K+1) float32; column 0 is `full_frame`.
  - `keyframe_mask`: (T,) bool, shared across ROIs.
  - `roi_ids`: ("full_frame", …).
  - `pixel_counts`: (K+1,).
  - `roi_bboxes`: (K+1, 4).
  - `avg_map`: (H, W) float32, full frame.
  - `metadata`: dict.

  Normalization divides each column by its own `pixel_counts`, not by the bbox area.
- Collect the per-frame int64 rows in a list and stack them at the end. `n_frames` from the probe can be wrong, so don't preallocate.
- `clean_trace` accepts 1-D or 2-D input (time on axis 0) with a 1-D mask. It interpolates per column using the shared good/bad indices.

### `cli.py`
- `--roi-masks PATH`, mutually exclusive with `--roi`:
  - **Directory:** each video's masks file is found by matching `*_{camera_key}_masks.npz` (reusing `camera_key`).
    - 0 matches → warning, and the video runs full-frame only.
    - More than 1 match → error.
  - **File:** allowed only when `--input` is a single video.
- Resolve and load every video's masks, and check them against `get_video_info` dimensions, **before** any processing. This fails fast, like `find_key_collisions`.
- Outputs per video (they replace the four `.npy` files):
  - `{stem}_motion_energy.npz` with keys `me`, `me_clean` (T,K+1) f4, `keyframe_mask` (T,), `frame_index` (T,) i8, `roi_ids` (<U), `pixel_counts`, `roi_bboxes`, `avg_map`. Written with uncompressed `np.savez` to a temp file in the output dir, then `os.replace` (atomic).
  - `{stem}_me_metadata.json` adds `output_version: 2`, `n_rois`, `roi_ids`, `roi_masks_path`, `roi_masks_sha256`.
  - `{stem}_roi_masks.npz`, a byte copy of the input masks, written only when masks were given (provenance).
  - `{stem}_motion_energy.csv` (`--format csv|both`), with columns `frame_index, is_keyframe, full_frame, full_frame_clean, <id>, <id>_clean, …`.
- `--format` choices become `npz | csv | both` (default `npz`).
- New flag `--viz-rois ID [ID ...]` selects which traces scroll in the MP4.

### `viz.py`
- **Trace PNG** (`{stem}_motion_energy.png`) becomes a single QC figure meant for scanning a batch of sessions in Code Ocean, not for analysis:
  - Top panel: the `full_frame` trace, raw in grey under clean, with keyframe ticks.
  - Bottom panel: an ROI × time heatmap, per-ROI z-scored and time-binned to ~2000 columns. ROI labels are shown when K ≤ 40.
  - This surfaces dead or flat ROIs, saturation, artifacts and dropouts at a glance, at any K.
- **Map PNG:** the full-frame `avg_map` with each ROI's outline (`ax.contour` of its mask at 0.5) and labels, using one colour per ROI.
- **MP4:** `render_motion_energy_video(video, result, rois=RoiSet|None, viz_rois=None, …)`.
  - Outlines are drawn once on the image axis.
  - The plotted traces are `full_frame` plus the first 4 ROIs by default, or `viz_rois`, with legend colours matching the outlines.
  - The raw trace stays a faint background line, drawn only for `full_frame`.
- Getting outlines needs masks, so the CLI passes the loaded masks. `RoiSet` keeps `bboxes` plus `flat_idx`, from which each cropped mask can be rebuilt. Add `RoiSet.mask(i) -> (bbox, cropped bool)`.

### Docs, version, downstream
- README: CLI flags, the Outputs table (new NPZ schema), and a "ROIs" section on the mask contract and filename matching.
- `docs/design-notes.md`: one entry covering "masks as the common pathway", "single-pass reduceat kernel" and "NPZ output (why, and its limits)".
- `docs/optimization-backlog.md`: update the map item (uint32 accumulator), and add the rectangle fast path, summed-area table and sparse matmul.
- Bump to 0.4.0. **[as built]** Deferred to a separate `chore:` PR, matching how this repo has done past bumps (#44).
- Update `examples/` notebook for the new outputs.
- Follow-ups in other repos, which this PR does not touch. **[done 2026-10-22]**
  - **`aind-motion-energy-capsule`:** pinned to `v0.4.0` (PR #12): README outputs and flags, and the example notebook on `MotionEnergyResult`. `code/run` needed no change; `--roi-masks` arrives via `"$@"`. A 5,000-frame test run matched the 0.3.0 trace shifted by one row. No capsule version is released; runs use the capsule's current state.
  - **`aind-motion-energy-batch`:** ROIs are set **per batch**, not per session (PR #20): a batch file `{"roi_masks_asset": "<id>", "sessions": [...]}` mounts one mask asset (one `*_<camera>_masks.npz` per camera) at `/root/capsule/data/roi_masks` in every run and passes `--roi-masks`. The skip-if-done check does not tell versions or ROI runs apart; reruns use `--force`.
  - **`kinematics_analysis`:** `code/me_results.py` reads 0.3.0 (`.npy`, pad a leading NaN) and 0.4.0 (`.npz`, already aligned) results by `output_version`, so neither is padded twice. `build_me_table.py`, `build_me_asset_map.py` and `check_leading_lost_frames.py` use it. ROI columns in the aligned table and an ROI argument to `fip_utils.load_me` are open in that repo's `code/fip_todo.md`.
  - The ROI annotator's box-ROI export (`export_rois.py` reading `x/y/w/h` instead of `bbox`) was left as is.
- **Known mismatch to document:** in flat layouts, the annotator's camera name (its stem heuristic picks the longest token) can differ from `camera_key` (the full stem). The directory-mode error message should tell the user to rename the mask file to `*_{camera_key}_masks.npz`, or to use file mode.

## Suggested PR sequence
1. Frame alignment (prerequisite).
2. `rois.py` + `compute.py` + `clean_trace` 2-D, with tests.
3. CLI + NPZ/CSV/metadata outputs, with tests and README.
4. Viz: QC trace PNG, map outlines, MP4 outlines and `--viz-rois`.
5. Docs, notebook and 0.4.0 bump; then the capsule and batch follow-ups.

## Verification
- Unit tests (keep the 100% coverage gate, `mypy --strict`, ruff):
  - **`rois`:** NPZ load round-trip, including a file written the way `export_rois.py` writes it; every validation error; `reduce` equal to a brute-force `(diff * mask).sum()` for random overlapping masks, polygons with holes, and a 1-pixel ROI; `from_box` equal to a slice sum.
  - **`compute`:** use the existing mocked `iter_luma_frames` pattern in `tests/test_compute.py`. Check K columns plus `full_frame`, per-ROI normalization, the shared keyframe mask, and `avg_map` at full-frame shape.
  - **CLI:** mask discovery with 0, 1 and many matches; `--roi` and `--roi-masks` being mutually exclusive; file mode rejected for directory input; a shape mismatch failing before any processing; NPZ keys and shapes; CSV header; metadata fields; the masks copy.
  - **`viz`:** smoke tests with K=3 and K=100 (labels off), plus an MP4 with `viz_rois`.
- **End-to-end:**
  1. Generate masks with the annotator on a real session, with boxes and a compound polygon.
  2. Run `uv run aind-motion-energy --input <session> --roi-masks <dir> --summary-plots --visualize`.
  3. Load the NPZ, and spot-check one ROI against a slow reference (a separate decode with that mask applied).
- **Benchmark** on the 15k-frame sample clip:
  - Compare full-frame only (the 0.3.0 baseline) against 1, 10 and 100 ROIs. Also sweep the total mask area at 0.5×, 2×, 4×, 10× and 20× the frame area, to set the guard thresholds.
  - Expect 100 typical ROIs to add little over baseline, and the uint32 map to come out faster than 0.3.0.
  - Record the numbers in the design notes.

## Real-session validation [as built]

`behavior_800886_2025-09-11_13-52-51`: flat layout, `bottom_camera.mp4` and `side_camera_right.mp4`, 720×540 at 1000 fps, 2.63M frames each. Used 10 ROIs drawn in the annotator (polygons and boxes) and a 15 s window per camera.

- **Traces:** every ROI matches a brute-force per-mask reference (max error 1e-7), and the outlines sit on the intended anatomy.
- **Speed:** 6 ROIs cost about 8% (1040 → 955 fps), so a full camera takes about 46 min.
- **Issues found and fixed here:**
  - Mask files named by the annotator (`…_BottomCamera_masks.npz`) didn't match `bottom_camera.mp4`. Matching now ignores case and separators, on whole `_`-separated parts, and mask files that match no video are listed.
  - Long ROI ids overflowed the legends and labels. Plots now drop the camera prefix all ROIs share.
  - Per-frame rows were held as a Python list of arrays, roughly 400 MB of overhead at 2.6M frames. They now go into a preallocated buffer that grows if needed.
- **Found in the annotator (not this repo):** `export_rois.build_mask` reads box geometry from top-level `x/y/w/h`, but the JSON stores it under `bbox`, so every box is exported as an empty mask. This library rejects empty masks with a clear error.

- **[as built] `frame_index` → `first_frame`.** The NPZ stores the starting frame number as one scalar instead of a per-row int64 array. Row i is frame `first_frame + i`. The array cost 21 MB per camera on a 2.6M-frame session, more than both full-frame traces together, and only repeated the row number in whole-video runs. The CSV keeps its `frame_index` column.

## Simplification pass before merge [as built]

Removed about 250 lines of library code without losing ROI features:

- `RoiSet` class → plain `dict` of name → bool mask.
- No bounding boxes.
- No size guards. The uint32/uint64 switch became a refusal of frames over 16.8 megapixels.
- `compute_motion_energy` returns frame-aligned traces, so `align_to_frames`, `n_diffs`, `n_output_frames` and `first_frame_fill` are gone.
- Outlines are drawn from full-size masks.
- No SHA-256; the copied masks file is the provenance.

The ROI heatmap and `--viz-rois` were kept.
