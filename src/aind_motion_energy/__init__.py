"""Motion energy computation from behavior videos for neuroscience pipelines."""

from .compute import align_to_frames, clean_trace, compute_motion_energy
from .io import get_video_info, iter_luma_frames
from .viz import render_motion_energy_video, save_summary_plots

__all__ = [
    "align_to_frames",
    "compute_motion_energy",
    "clean_trace",
    "get_video_info",
    "iter_luma_frames",
    "render_motion_energy_video",
    "save_summary_plots",
]
