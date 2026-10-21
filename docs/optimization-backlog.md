# Optimization backlog

Candidate speed optimizations not yet implemented. See `docs/design-notes.md`
for *why* decisions; this file is *what's left to try*.

## Faster multi-ROI reduction

The ROI reduction in `compute_motion_energy` costs ~2 ns per masked pixel per frame; 100
ROIs of 3k px cut end-to-end throughput to ~55% of full-frame-only on the
sample clip. Candidates, each measured on 720×540 / 100 ROIs / 300k px:

- **Rectangle fast path:** exact boxes summed by slicing, `diff[y:y+h, x:x+w]`
  — 0.47 ms vs 0.64 ms for the general kernel. Small win; only worth it if box
  ROIs dominate in practice.
- **Summed-area table:** one cumulative-sum pass then O(1) per box — 3.9 ms at
  this size, so slower unless there are thousands of boxes.
- **Overlap reduction with decode:** run the reduction for frame *i* while
  frame *i+1* decodes (NumPy releases the GIL on large ops). Largest potential
  win; adds threading complexity.
- **`scipy.sparse` mask matrix × diff:** not measured (scipy isn't a
  dependency); would need float64 to stay exact.

## `--no-map` flag to skip the average motion map

The per-frame full-frame float64 `map_accumulator += diff` in
`compute_motion_energy` is ~45% of runtime (measured on the 15k-frame sample
clip) — the single biggest compute hog at full resolution.

- **Change:** add `compute_map: bool = True` to `compute_motion_energy` and a
  `--no-map` flag in `cli.py`; when off, skip the accumulation, the final
  divide, and the `avg_map` array in the NPZ (and the avg-map summary PNG).
- **Update (multi-ROI):** the accumulator is now uint32 instead of float64,
  ~15% cheaper per frame, so the expected win is somewhat smaller than below.
- **Expected win:** ~1.8x at full res (caps near the diff+sum floor; the `diff`
  itself is still needed for the ME sum). Pixel-preserving — unlike ROI/downscale
  it does not change the motion-energy trace, only drops the map output.
- **Cost:** loses `avg_map.npy` + the average-motion-map summary plot. Fine for
  batch throughput where only the trace is needed.
- **Status:** not implemented. Today the map is always computed and saved.
