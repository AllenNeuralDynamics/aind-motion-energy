# Downstream changes for the multi-ROI release (0.4.0)

What has to change outside this repo before 0.4.0 outputs are used. Checked
against each repo's state on 2026-10-21.

## What changed in the outputs

- The four `.npy` files are replaced by one `{stem}_motion_energy.npz`. The
  0.3.0 `motion_energy_clean.npy` trace is now `npz["me_clean"][:, 0]`
  (column `full_frame`). ROI traces are the other columns, named by
  `npz["roi_ids"]`.
- **Padding moved into the library.** Saved traces have one value per frame
  (length N, row i = frame i, row 0 = NaN). 0.3.0 saved N−1 values, and readers
  added the leading NaN themselves. **Readers must stop padding**, or every
  trace shifts one frame late.
- The metadata key `n_me_frames` is now `n_diffs` (still N−1). The rename is
  deliberate: code that pads when `n_me_frames == n_frames_decoded - 1` now
  fails with a `KeyError` instead of silently padding twice. Use
  `n_output_frames`, which equals `n_frames_decoded` for a whole-video run.
- `--format {npy,csv,both}` is replaced by a `--csv` flag (the NPZ is always
  written). Callers passing `--format csv` or `--format both` must switch to `--csv`.
- The Python API changed: `compute_motion_energy` returns a `MotionEnergyResult`
  (not a 4-tuple), and takes `rois=` instead of `roi=`.

## kinematics_analysis

| File | Change |
|---|---|
| `code/build_me_table.py` | Download `<source>_motion_energy.npz` instead of `_motion_energy_clean.npy`; take `me_clean[:, roi_ids.index("full_frame")]`. **Delete the NaN padding step** and the `n_me_frames`/`padded` logic. Require `len(me) == n_frames_decoded` (`n_output_frames`) and refuse otherwise. The `padded` index column can go, or be recorded as always False. |
| `code/check_leading_lost_frames.py` | Same load change. **Remove `np.concatenate([[np.nan], me])`** (line ~92). This script has no length check, so leaving the pad in would shift ME one frame late silently. |
| `code/build_me_asset_map.py` | Replace the check `n_me_frames == n_frames_decoded - 1` (line ~115) with `n_output_frames == n_frames_decoded`, and look for `_motion_energy.npz` alongside `_me_metadata.json`. |
| `code/fip_me_aligned_table_plan.md` | Update the "Padding" decision and step 2.6 ("pad a leading NaN…") to say the library now pads. |

Not affected: `fip_utils.load_me` / `motion_energy_to_session`, `men_*`,
`cue_me_maps.py`, `make_me_example_clip.py`. They read the aligned Parquet
table, which keeps its layout (row i = frame i, NaN at row 0). The table
only needs rebuilding if ROI columns are wanted, or ME is recomputed with 0.4.0.

Optional, later: add ROI columns to the Parquet table (one per `roi_ids`
entry), so `load_me` can return a chosen ROI instead of only the whole frame.

## aind-motion-energy-capsule

| File | Change |
|---|---|
| `environment/` pin | Pin `aind-motion-energy` 0.4.0 (after release). |
| `README.md` outputs table | Replace the four `.npy` rows with `{stem}_motion_energy.npz` and `{stem}_roi_masks.npz`; describe the frame alignment and NaN row 0. Add `--roi-masks` and `--viz-rois` to the list of flags passed via `"$@"`, and replace `--format` with `--csv` (README lines 41 and 54). |
| Data mounts | For ROI runs, mount the session's mask files (a folder of `*_<camera>_masks.npz`) and pass `--roi-masks <that folder>` via `"$@"`. `code/run` itself needs no change. |
| `examples/example_motion_energy.ipynb` | `me, keyframe_mask, avg_map, meta = compute_motion_energy(...)` → `result = compute_motion_energy(...)`, then `result.me[:, 0]`, `result.keyframe_mask`, `result.avg_map`, `result.metadata`. |

## aind-motion-energy-batch-capsule

| File | Change |
|---|---|
| `code/run_capsule.py` | For ROI runs: find each session's mask asset, mount it next to the raw asset (`DataAssetsRunParam`), and add `--roi-masks /root/capsule/data/<mask mount>` to `ME_PARAMETERS`. Sessions without masks still run whole-frame only. |
| Result checks / docs | Any check that a result is complete should look for `_motion_energy.npz` + `_me_metadata.json`, not the `.npy` files (`docs/batch-plan.md` step "files present"). |
| Coexistence | 0.3.0 results already in `s3://aind-scratch-data/matt.becker/motion_energy/` keep the old files. Readers in `kinematics_analysis` need to know which version produced a result: `output_version` in the metadata is `2` for 0.4.0 and absent for 0.3.0. |

## ROI annotator app (generate_roi_html.py / export_rois.py)

| File | Change |
|---|---|
| `export_rois.py` `build_mask` | **Bug:** box ROIs are read from top-level `x/y/w/h`, but the JSON stores them under `bbox`, so every box exports as an empty mask. Read `geom = roi.get("bbox", roi)` and use `geom["x"]` and so on. (`draw_roi_overlay` has the same bug.) |
| Browser **↓ NPZ** export | Untested against this library, and reportedly not working in the app. Check that `load_roi_masks` accepts its output once it works. |

## Order

1. Merge this branch into `dev`, then release 0.4.0 (version bump PR).
2. Fix the annotator's box bug, so mask files are trustworthy.
3. Update the ME capsule (pin, README, notebook), then the batch launcher (mounts, `--roi-masks`).
4. Update `kinematics_analysis` readers before reading any 0.4.0 result. Until then, they will fail loudly on the missing `.npy` files, which is safe.
