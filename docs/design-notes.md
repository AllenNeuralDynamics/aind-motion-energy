# Design notes

Short rationale for non-obvious technical decisions. One entry per decision;
newest first. For *how to use* the library, see the README — this file is *why*.

## ROIs are binary masks, reduced in one pass per frame

- **Masks are the only ROI representation the library consumes.** The ROI
  annotator rasterizes boxes, polygons and compound (add/subtract) shapes into
  `masks (H, W, K)` + `roi_ids`; `rois.py` never sees geometry. One code path
  covers every shape, and rasterization rules live in one place (the annotator).
  `--roi X Y W H` is just `RoiSet.from_box`.
- **One decode, one diff, one reduction call.** The full-frame diff is needed
  anyway (full-frame column + average map), so ROIs only add the reduction
  `np.add.reduceat(diff.ravel()[flat_idx], offsets, dtype=uint32)`, with
  `flat_idx` the concatenated mask pixel indices. No Python loop over ROIs; cost
  is proportional to total masked pixels, not ROI count. That's why the size
  guard is on total area (warn > 2× frame, refuse > 20×), not on K.
- **Kernel choice, measured** (720×540, 100 ROIs, 300k masked px, per frame):
  `take` + `reduceat(int64)` 2.6 ms; bbox crop + `sum(where=mask)` 2.5 ms;
  fancy index (`intp`) + `reduceat(int64)` 1.0 ms; fancy index +
  `reduceat(uint32)` **0.64 ms** (chosen). `take` and narrow index dtypes are
  ~4× slower than native-`intp` fancy indexing; int64 accumulation pays a cast
  loop that uint32 avoids. uint32 is exact for any mask under 2³²/255 ≈ 16.8M px;
  larger masks switch to uint64 automatically. End-to-end on the 15k-frame
  sample clip, no ROIs matches 0.3.0 throughput, and 100 such ROIs run at ~55%
  of it (decode dominates the rest).
- **Full-frame map accumulates in uint32**, not float64: exact (overflow only
  past ~16.8M frames) and ~15% cheaper per frame.

## Outputs: one uncompressed NPZ per video

`{stem}_motion_energy.npz` holds every numeric output (traces `(T, 1+K)`,
keyframe mask, frame index, ROI ids/pixel counts/bboxes, average map).

- **Why NPZ:** same format as the annotator's mask export; one file per video
  keeps `me` and the `roi_ids` that name its columns from drifting apart (the
  risk with separate `.npy` files); no new dependency; float32/bool kept exactly.
- **Why uncompressed:** deflate barely shrinks noisy float traces and slows
  both writes and reads.
- **Limits accepted:** no memory-mapping or partial reads of one array (a
  1 h, 500 fps, 100-ROI trace is ~720 MB in RAM per array); no attributes (the
  JSON sidecar carries metadata); not natively readable in MATLAB (unzip +
  npy-matlab). The wide CSV and a future NWB packaging step cover non-Python
  consumers. Written via temp file + `os.replace` so a crash never leaves a
  half-written ZIP (whose index sits at the end).

## Decode via PyAV (libav*), not the ffmpeg CLI or OpenCV

`io.py` decodes frames with PyAV. PyAV binds the ffmpeg C libraries
(`libavcodec`/`libavformat`) in-process — it is a sibling of the `ffmpeg`
binary (which is itself just an app on those libraries), not a wrapper above it.

- **vs OpenCV:** OpenCV also sits on libavcodec but adds an opaque layer with
  opinionated defaults (forced BGR, silent seek-to-keyframe) and hides which
  decoder it chose. PyAV keeps ffmpeg-level explicitness (pixel format, planes,
  per-frame keyframe flag, codec name).
- **vs shelling out to the ffmpeg binary:** a raw-pixel pipe carries pixels
  only — we'd lose per-frame metadata and need a second `ffprobe` pass matched
  by timestamp to recover keyframe flags, plus subprocess + serialization
  overhead. PyAV yields pixels **and** the keyframe flag **and** codec name in
  one in-process decode pass (see `iter_luma_frames`), so it's actually *fewer*
  layers for "frames into Python," despite the binary feeling more direct.
- **Caveat (pragmatic split):** upfront metadata still uses the `ffprobe`
  binary via `aind_video_utils.probe` in `get_video_info` — one cheap call where
  the process boundary doesn't matter. PyAV is reserved for the hot per-frame loop.
- **Version risk:** `pyproject.toml` floors `av>=17.1.0`; exact version is frozen
  in `uv.lock`. PyAV wheels statically bundle their own libav*, so installs don't
  use the system ffmpeg. Decoder upgrades are very unlikely to change decoded
  pixel values; the real risk of an unpinned bump is API/attribute drift, which
  the lockfile prevents.
