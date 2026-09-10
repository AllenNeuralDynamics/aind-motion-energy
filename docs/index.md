# aind-motion-energy

Motion energy computation from behavior videos for neuroscience pipelines.

Motion energy — the sum of absolute frame-to-frame differences — is a standard,
markerless summary of how much an animal moved in each frame of a behavior
video (e.g. Musall et al. 2019). This library computes it from a video file or a
directory of videos, either through a CLI or as a Python API, and is deployed to
Code Ocean through a [thin capsule wrapper](#related-repos).

Frames are decoded in-process with PyAV rather than through the `ffmpeg` binary
or OpenCV, which keeps the per-frame keyframe flag available — see
[Design notes](design-notes.md) for why that matters.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
uv sync --dev
```

Or, to install just the library — the `[viz]` extra pulls in `matplotlib`, which
the plotting and video-rendering outputs need:

```bash
pip install aind-motion-energy
pip install "aind-motion-energy[viz]"
```

## Quickstart

### CLI

```bash
uv run aind-motion-energy --input data/my_video.mp4 --output results/ --summary-plots
```

`--input` takes a single video or a directory, searched recursively. Every run
writes a raw trace, a cleaned trace, a keyframe mask, a per-pixel average motion
map, and a JSON metadata sidecar per video. The
[README](https://github.com/AllenNeuralDynamics/aind-motion-energy#cli-usage)
carries the full flag reference and the output-file naming convention, including
the camera-keying heuristic the capsule depends on.

### Python

```python
from pathlib import Path

from aind_motion_energy import clean_trace, compute_motion_energy

motion_energy, keyframe_mask, avg_map, metadata = compute_motion_energy(
    Path("data/my_video.mp4"),
    roi=(100, 50, 400, 300),  # (x, y, w, h); omit for the full frame
)

# Raw motion energy is never altered — compression "pops" at H.264/HEVC
# keyframes are flagged in keyframe_mask instead. Clean a copy for plotting
# or regression:
trace = clean_trace(motion_energy, keyframe_mask, method="interpolate")
```

Every public function is documented in the [API reference](reference/index.md).

## Examples

[`examples/ibl_vs_aind_motion_energy_comparison.ipynb`](https://github.com/AllenNeuralDynamics/aind-motion-energy/blob/main/examples/ibl_vs_aind_motion_energy_comparison.ipynb)
compares this implementation against the IBL motion-energy pipeline on the same
footage.

## Related repos

The pipeline spans three repositories:

| Repo | Role |
|---|---|
| [`aind-motion-energy`](https://github.com/AllenNeuralDynamics/aind-motion-energy) | Python library + CLI — the algorithm. This repo. |
| [`aind-motion-energy-capsule`](https://github.com/AllenNeuralDynamics/aind-motion-energy-capsule) | Code Ocean capsule; a thin shell that forwards its arguments to this library's CLI. |
| [`aind-motion-energy-batch`](https://github.com/AllenNeuralDynamics/aind-motion-energy-batch) | Code Ocean launcher; fans out one capsule run per session. |

## Contributing

Branch off `dev`, use [conventional commit](https://www.conventionalcommits.org/)
PR titles, and link the PR to an issue. `uv run ruff check`, `uv run pytest`, and
`uv run mkdocs build --strict` all have to pass. See
[CONTRIBUTING.md](https://github.com/AllenNeuralDynamics/aind-motion-energy/blob/main/CONTRIBUTING.md)
and the [standards compliance plan](standards-compliance.md).
