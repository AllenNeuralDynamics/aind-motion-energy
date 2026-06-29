# Optimization backlog

Candidate speed optimizations not yet implemented. See `docs/design-notes.md`
for *why* decisions; this file is *what's left to try*.

## `--no-map` flag to skip the average motion map

The per-frame full-frame float64 `map_accumulator += diff` in
`compute_motion_energy` is ~45% of runtime (measured on the 15k-frame sample
clip) — the single biggest compute hog at full resolution.

- **Change:** add `compute_map: bool = True` to `compute_motion_energy` and a
  `--no-map` flag in `cli.py`; when off, skip the accumulation, the final
  divide, and the `..._motion_energy_map.npy` save (and the avg-map summary PNG).
- **Expected win:** ~1.8x at full res (caps near the diff+sum floor; the `diff`
  itself is still needed for the ME sum). Pixel-preserving — unlike ROI/downscale
  it does not change the motion-energy trace, only drops the map output.
- **Cost:** loses `avg_map.npy` + the average-motion-map summary plot. Fine for
  batch throughput where only the trace is needed.
- **Status:** not implemented. Today the map is always computed and saved.
