"""Static summary plots and scrolling motion-energy video rendering."""

from pathlib import Path
from typing import Any

import av
import numpy as np

from .io import iter_luma_frames

_VIZ_IMPORT_HINT = (
    "Rendering motion-energy videos requires matplotlib. "
    "Install it with: pip install 'aind-motion-energy[viz]'"
)


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_VIZ_IMPORT_HINT) from exc
    return plt


def save_summary_plots(
    output_dir: Path,
    stem: str,
    me: np.ndarray,
    me_clean: np.ndarray,
    avg_map: np.ndarray,
    *,
    fps: float,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Save two static PNG summaries for one video.

    1. ``{stem}_motion_energy.png`` — the motion-energy timeseries: the raw trace
       as a faint gray background line and the cleaned trace as a bold steelblue
       line on top, over source time in seconds.
    2. ``{stem}_motion_energy_map.png`` — a heatmap of the per-pixel average
       absolute frame difference (the spatial map of where motion occurred).

    Requires matplotlib (the ``[viz]`` extra).

    Parameters
    ----------
    output_dir : Path
        Directory to save the PNGs into.
    stem : str
        Filename stem shared by both PNGs and used as the plot title.
    me : numpy.ndarray
        Raw motion-energy trace.
    me_clean : numpy.ndarray
        Cleaned motion-energy trace (see `clean_trace`).
    avg_map : numpy.ndarray
        Per-pixel average absolute frame difference, shape ``(H, W)``.
    fps : float
        Source video frame rate, used to convert sample index to seconds.
    dpi : int, optional
        Resolution of the saved PNGs, by default 150.

    Returns
    -------
    tuple[Path, Path]
        Paths to the timeseries PNG and the heatmap PNG.
    """
    plt = _import_pyplot()

    output_dir = Path(output_dir)
    t = np.arange(len(me)) / fps

    trace_path = output_dir / f"{stem}_motion_energy.png"
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, me, lw=0.4, color="0.78", alpha=0.8)
    ax.plot(t, me_clean, lw=0.6, color="steelblue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Motion energy")
    ax.set_title(stem)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(trace_path, dpi=dpi)
    plt.close(fig)

    map_path = output_dir / f"{stem}_motion_energy_map.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_map, cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean abs diff (per pixel)")
    ax.set_title(f"{stem} — avg motion map")
    fig.tight_layout()
    fig.savefig(map_path, dpi=dpi)
    plt.close(fig)

    return trace_path, map_path


def _build_trace_figure(
    plt: Any, dpi: int, trace_t: np.ndarray, trace: np.ndarray, raw: np.ndarray | None
) -> tuple[Any, Any, Any, Any]:
    """Build the image+trace figure used as the render_motion_energy_video canvas.

    Parameters
    ----------
    plt : module
        The imported ``matplotlib.pyplot`` module.
    dpi : int
        Figure resolution.
    trace_t : numpy.ndarray
        Time values (seconds) for each point in `trace`.
    trace : numpy.ndarray
        Cleaned motion-energy trace to plot.
    raw : numpy.ndarray or None
        Raw motion-energy trace to plot as a faint background line, if given.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax_img : matplotlib.axes.Axes
        Axes for the video frame image.
    ax_plot : matplotlib.axes.Axes
        Axes for the scrolling trace plot.
    cursor : matplotlib.lines.Line2D
        Vertical line marking the current playback time.
    """
    fig, (ax_img, ax_plot) = plt.subplots(
        2, 1, figsize=(8, 7), dpi=dpi, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_img.axis("off")
    if raw is not None:
        ax_plot.plot(trace_t, raw, lw=0.5, color="0.78", alpha=0.8, zorder=1)
    ax_plot.plot(trace_t, trace, lw=0.7, color="steelblue", zorder=2)
    cursor = ax_plot.axvline(float(trace_t[0]) if len(trace_t) else 0.0, lw=0.8, color="0.5")
    ax_plot.set_xlabel("Time (s)")
    ax_plot.set_ylabel("Motion energy")
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    if len(trace):
        upper = float(raw.max()) if raw is not None else float(trace.max())
        lower = float(raw.min()) if raw is not None else float(trace.min())
        pad = 0.05 * ((upper - lower) or 1.0)
        ax_plot.set_ylim(lower - pad, upper + pad)
    fig.tight_layout()
    return fig, ax_img, ax_plot, cursor


def _capture_rgb_frame(fig: Any) -> np.ndarray:
    """Rasterize the current figure canvas to an even-dimensioned RGB array.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to rasterize.

    Returns
    -------
    numpy.ndarray
        RGB array with even height and width (required by the H.264 encoder).
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(buf[..., :3])
    h = rgb.shape[0] - (rgb.shape[0] % 2)
    w = rgb.shape[1] - (rgb.shape[1] % 2)
    return rgb[:h, :w]


def _open_video_writer(
    output_path: Path, out_fps: float, width: int, height: int
) -> tuple[Any, Any]:
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("libx264", rate=int(round(out_fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return container, stream


def _finalize_video_writer(container: Any, stream: Any) -> None:
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def render_motion_energy_video(
    video_path: Path,
    trace: np.ndarray,
    *,
    fps_source: float,
    output_path: Path,
    raw_trace: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    window_seconds: float = 3.0,
    out_fps: float = 60.0,
    stride: int = 1,
    dpi: int = 100,
) -> Path:
    """Render an MP4 of the footage with a synced, scrolling motion-energy plot.

    The grayscale luma frame is shown on top and the motion-energy plot scrolls
    below it with a thin cursor. The displayed pixels are the exact Y-plane array
    motion energy was computed on (identical to native for monochrome cameras).

    `trace` has one fewer entry than the number of frames. The x-axis is source
    time in seconds, independent of `out_fps`. `out_fps` only sets playback
    speed of the output file (no frames are dropped).

    Parameters
    ----------
    video_path : Path
        Path to the source video.
    trace : numpy.ndarray
        Cleaned (interpolated) motion-energy trace to plot.
    fps_source : float
        Frame rate of the source video, used to convert frame index to time.
    output_path : Path
        Path to write the rendered MP4 to.
    raw_trace : numpy.ndarray or None, optional
        Unmodified motion-energy trace including keyframe artifact spikes.
        When given, both traces are drawn: raw as a faint gray background
        line and clean as a bold steelblue line on top. This lets a viewer
        see the encoding spikes at their true amplitude and position,
        confirming the clean trace tracks genuine motion.
    roi : tuple[int, int, int, int] or None, optional
        Region of interest as ``(x, y, w, h)`` in pixels.
    start_frame : int or None, optional
        First frame to render, inclusive.
    end_frame : int or None, optional
        Last frame to render, exclusive.
    window_seconds : float, optional
        Width of the scrolling plot window in seconds, by default 3.0.
    out_fps : float, optional
        Playback frame rate of the output file, by default 60.0.
    stride : int, optional
        Render every Nth frame; values > 1 trade temporal resolution for
        render time / file size, by default 1 (lossless).
    dpi : int, optional
        Figure resolution, by default 100.

    Returns
    -------
    Path
        Path to the rendered MP4 (same as `output_path`).

    Raises
    ------
    ValueError
        If `stride` is less than 1.
    """
    plt = _import_pyplot()

    output_path = Path(output_path)
    trace = np.asarray(trace, dtype=np.float32)
    raw = np.asarray(raw_trace, dtype=np.float32) if raw_trace is not None else None
    start = start_frame or 0
    if stride < 1:
        raise ValueError(f"stride must be >= 1 (got {stride})")

    # trace[m] is the diff ending at frame (start + m + 1); place it at that time.
    trace_t = (start + np.arange(len(trace)) + 1) / fps_source
    fig, ax_img, ax_plot, cursor = _build_trace_figure(plt, dpi, trace_t, trace, raw)

    img_artist = None
    container = None
    stream = None
    try:
        for j, (frame, _is_key) in enumerate(
            iter_luma_frames(video_path, roi=roi, start_frame=start_frame, end_frame=end_frame)
        ):
            if j % stride != 0:
                continue
            t_now = (start + j) / fps_source

            if img_artist is None:
                img_artist = ax_img.imshow(frame, cmap="gray", vmin=0, vmax=255, aspect="equal")
            else:
                img_artist.set_data(frame)
            ax_plot.set_xlim(t_now - window_seconds, t_now)
            cursor.set_xdata([t_now, t_now])

            rgb = _capture_rgb_frame(fig)
            if container is None:
                container, stream = _open_video_writer(
                    output_path, out_fps, rgb.shape[1], rgb.shape[0]
                )

            vframe = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(vframe):
                container.mux(packet)

        if container is not None:
            _finalize_video_writer(container, stream)
    finally:
        plt.close(fig)

    return output_path
