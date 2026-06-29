# Design notes

Short rationale for non-obvious technical decisions. One entry per decision;
newest first. For *how to use* the library, see the README — this file is *why*.

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
