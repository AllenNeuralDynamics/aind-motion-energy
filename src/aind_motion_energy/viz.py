"""Static summary plots and scrolling motion-energy video rendering."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np

from .io import iter_luma_frames
from .rois import FULL_FRAME_ID

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


FULL_FRAME_COLOR = "steelblue"
# Heatmap time bins: enough to see structure, few enough to render quickly.
HEATMAP_MAX_BINS = 2000
HEATMAP_MAX_LABELS = 40
DEFAULT_VIZ_TRACES = 5


def _as_columns(trace: np.ndarray) -> np.ndarray:
    """View a ``(T,)`` or ``(T, K)`` trace as float32 ``(T, K)``."""
    arr = np.asarray(trace, dtype=np.float32)
    return arr[:, None] if arr.ndim == 1 else arr


def _column_ids(n_columns: int, roi_ids: Sequence[str] | None) -> list[str]:
    if roi_ids is not None:
        if len(roi_ids) != n_columns:
            raise ValueError(f"{len(roi_ids)} roi_ids for {n_columns} trace columns")
        return list(roi_ids)
    return [FULL_FRAME_ID] + [f"roi{i}" for i in range(1, n_columns)]


def _short_labels(ids: Sequence[str]) -> list[str]:
    """Display labels for ROI ids, minus the ``<camera>_`` prefix they all share.

    The annotator names ROIs ``<Camera>_<name>_<type>``, which is too long to
    fit in legends and on outlines; the camera is already in the plot title.
    Only affects plots, never saved ids.
    """
    if len(ids) < 2:
        return list(ids)
    cut = os.path.commonprefix(list(ids)).rfind("_") + 1
    short = [i[cut:] for i in ids]
    return short if all(short) else list(ids)


def _roi_color(plt: Any, roi_index: int) -> Any:
    """Colour of ROI ``roi_index`` (0-based, excluding full_frame), shared by every plot.

    Cycles tab10 without its first (blue) entry, which is too close to the
    steelblue full-frame trace.
    """
    return plt.get_cmap("tab10")(1 + roi_index % 9)


def _draw_roi_outlines(plt: Any, ax: Any, rois: dict[str, np.ndarray], lw: float = 1.0) -> None:
    """Outline each ROI's mask on an image axis and label it at its top-left corner."""
    from matplotlib import patheffects

    for i, (mask, label) in enumerate(zip(rois.values(), _short_labels(list(rois)))):
        rows, cols = np.nonzero(mask)
        x, y = int(cols.min()), int(rows.min())
        color = _roi_color(plt, i)
        # A dark halo keeps outlines and labels legible on any background.
        halo = [patheffects.withStroke(linewidth=lw + 1.5, foreground="black")]
        # Contour coordinates are pixel centres, matching imshow.
        cs = ax.contour(mask.astype(np.float32), levels=[0.5], colors=[color], linewidths=lw)
        cs.set(path_effects=halo)
        ax.text(x, y - 2, label, color=color, fontsize=7, va="bottom", path_effects=halo)


def _zscore_binned(values: np.ndarray, max_bins: int) -> np.ndarray:
    """Per-column z-score of ``(T, K)`` values, NaN-mean-pooled to at most `max_bins` rows.

    NaNs (e.g. frame 0, which has no preceding frame) are ignored; a bin with
    no finite values stays NaN and renders blank.
    """
    n = len(values)
    n_bins = max(1, min(n, max_bins))
    starts = np.linspace(0, n, n_bins + 1).astype(int)[:-1]
    valid = np.isfinite(values)
    sums = np.add.reduceat(np.where(valid, values, 0.0), starts, axis=0)
    counts = np.add.reduceat(valid, starts, axis=0, dtype=np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = sums / counts
    finite = np.isfinite(pooled)
    n_finite = np.maximum(finite.sum(axis=0), 1)
    mean = np.where(finite, pooled, 0.0).sum(axis=0) / n_finite
    var = np.where(finite, (pooled - mean) ** 2, 0.0).sum(axis=0) / n_finite
    std = np.sqrt(var)
    std[std == 0] = 1.0
    return np.asarray((pooled - mean) / std, dtype=np.float32)


def _plot_full_frame(ax: Any, t: np.ndarray, raw: np.ndarray, clean: np.ndarray) -> None:
    ax.plot(t, raw, lw=0.4, color="0.78", alpha=0.8)
    ax.plot(t, clean, lw=0.6, color=FULL_FRAME_COLOR)
    ax.set_ylabel("Motion energy\n(full frame)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_roi_heatmap(
    fig: Any, ax: Any, t: np.ndarray, me_clean: np.ndarray, ids: list[str]
) -> None:
    z = _zscore_binned(me_clean, HEATMAP_MAX_BINS)
    t_end = float(t[-1]) if len(t) else 0.0
    im = ax.imshow(
        z.T,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
        extent=(float(t[0]) if len(t) else 0.0, t_end, len(ids) - 0.5, -0.5),
        vmin=-3,
        vmax=3,
    )
    if len(ids) <= HEATMAP_MAX_LABELS:
        ax.set_yticks(range(len(ids)), _short_labels(ids), fontsize=6)
    else:
        ax.set_ylabel(f"{len(ids)} ROIs")
    # Inset colorbar, so the heatmap keeps the same width (time axis) as the trace above.
    fig.colorbar(im, cax=ax.inset_axes((1.005, 0.0, 0.01, 1.0)), label="z-score")


def save_summary_plots(
    output_dir: Path,
    stem: str,
    me: np.ndarray,
    me_clean: np.ndarray,
    avg_map: np.ndarray,
    *,
    fps: float,
    roi_ids: Sequence[str] | None = None,
    keyframe_mask: np.ndarray | None = None,
    rois: dict[str, np.ndarray] | None = None,
    dpi: int = 150,
) -> tuple[Path, Path]:
    """Save two static PNG summaries for one video, meant for quick QC.

    1. ``{stem}_motion_energy.png`` — the full-frame motion-energy trace (raw
       as a faint gray line under the cleaned trace, keyframe-contaminated
       frames ticked along the bottom). When there are ROIs, a heatmap below
       it shows every ROI's cleaned trace over time, each z-scored to its own
       range, so a dead, flat or artifact-ridden ROI stands out at any ROI
       count.
    2. ``{stem}_motion_energy_map.png`` — a heatmap of the per-pixel average
       absolute frame difference, with each ROI outlined and labelled.

    Requires matplotlib (the ``[viz]`` extra).

    Parameters
    ----------
    output_dir : Path
        Directory to save the PNGs into.
    stem : str
        Filename stem shared by both PNGs and used as the plot title.
    me : numpy.ndarray
        Raw motion energy, shape ``(T,)`` or ``(T, 1 + K)`` with the
        full-frame trace in column 0.
    me_clean : numpy.ndarray
        Cleaned motion energy (see `clean_trace`), same shape as `me`.
    avg_map : numpy.ndarray
        Per-pixel average absolute frame difference, shape ``(H, W)``.
    fps : float
        Source video frame rate, used to convert sample index to seconds.
    roi_ids : Sequence[str] or None, optional
        Name of each column of `me`; defaults to ``full_frame, roi1, ...``.
    keyframe_mask : numpy.ndarray or None, optional
        Bool array of shape ``(T,)``; flagged frames are ticked on the trace.
    rois : dict[str, numpy.ndarray] or None, optional
        ROIs to outline on the map.
    dpi : int, optional
        Resolution of the saved PNGs, by default 150.

    Returns
    -------
    tuple[Path, Path]
        Paths to the timeseries PNG and the heatmap PNG.
    """
    plt = _import_pyplot()

    output_dir = Path(output_dir)
    raw, clean = _as_columns(me), _as_columns(me_clean)
    ids = _column_ids(raw.shape[1], roi_ids)
    t = np.arange(len(raw)) / fps

    trace_path = output_dir / f"{stem}_motion_energy.png"
    if raw.shape[1] > 1:
        fig, (ax, ax_heat) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )
        _plot_roi_heatmap(fig, ax_heat, t, clean[:, 1:], ids[1:])
        ax_heat.set_xlabel("Time (s)")
    else:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_xlabel("Time (s)")
    _plot_full_frame(ax, t, raw[:, 0], clean[:, 0])
    if keyframe_mask is not None and keyframe_mask.any():
        ax.plot(t[keyframe_mask], np.zeros(int(keyframe_mask.sum())), "|", color="firebrick", ms=4)
    ax.set_title(stem)
    fig.tight_layout()
    fig.savefig(trace_path, dpi=dpi)
    plt.close(fig)

    map_path = output_dir / f"{stem}_motion_energy_map.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_map, cmap="gray", aspect="equal")
    if rois is not None:
        _draw_roi_outlines(plt, ax, rois, lw=0.8)
    plt.colorbar(im, ax=ax, label="Mean abs diff (per pixel)")
    ax.set_title(f"{stem}\navg motion map", fontsize=9)
    fig.tight_layout()
    fig.savefig(map_path, dpi=dpi)
    plt.close(fig)

    return trace_path, map_path


def _build_trace_figure(
    plt: Any,
    dpi: int,
    trace_t: np.ndarray,
    traces: list[tuple[str, np.ndarray, Any]],
    raw: np.ndarray | None,
) -> tuple[Any, Any, Any, Any]:
    """Build the image+trace figure used as the render_motion_energy_video canvas.

    Parameters
    ----------
    plt : module
        The imported ``matplotlib.pyplot`` module.
    dpi : int
        Figure resolution.
    trace_t : numpy.ndarray
        Time values (seconds) for each trace point.
    traces : list[tuple[str, numpy.ndarray, Any]]
        ``(label, cleaned trace, colour)`` for each trace to plot.
    raw : numpy.ndarray or None
        Raw full-frame trace to plot as a faint background line, if given.

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
    for label, trace, color in traces:
        ax_plot.plot(trace_t, trace, lw=0.7, color=color, zorder=2, label=label)
    if len(traces) > 1:
        ax_plot.legend(fontsize=6, loc="upper left", ncol=min(len(traces), 3), frameon=False)
    cursor = ax_plot.axvline(float(trace_t[0]) if len(trace_t) else 0.0, lw=0.8, color="0.5")
    ax_plot.set_xlabel("Time (s)")
    ax_plot.set_ylabel("Motion energy")
    ax_plot.spines["top"].set_visible(False)
    ax_plot.spines["right"].set_visible(False)
    plotted = [tr for _, tr, _ in traces] + ([raw] if raw is not None else [])
    finite = np.concatenate([a[np.isfinite(a)] for a in plotted]) if plotted else np.array([])
    if finite.size:
        upper, lower = float(finite.max()), float(finite.min())
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


def _select_traces(
    plt: Any, clean: np.ndarray, ids: list[str], viz_rois: Sequence[str] | None
) -> list[tuple[str, np.ndarray, Any]]:
    """Pick the ``(label, trace, colour)`` columns to plot in the video."""
    if viz_rois is None:
        chosen = list(range(min(len(ids), DEFAULT_VIZ_TRACES)))
    else:
        unknown = [r for r in viz_rois if r not in ids]
        if unknown:
            raise ValueError(f"unknown viz_rois {unknown}; available: {ids}")
        chosen = [ids.index(r) for r in viz_rois]
    labels = [FULL_FRAME_ID, *_short_labels(ids[1:])]
    return [
        (labels[c], clean[:, c], FULL_FRAME_COLOR if c == 0 else _roi_color(plt, c - 1))
        for c in chosen
    ]


def render_motion_energy_video(
    video_path: Path,
    trace: np.ndarray,
    *,
    fps_source: float,
    output_path: Path,
    raw_trace: np.ndarray | None = None,
    roi_ids: Sequence[str] | None = None,
    rois: dict[str, np.ndarray] | None = None,
    viz_rois: Sequence[str] | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    window_seconds: float = 3.0,
    out_fps: float = 60.0,
    stride: int = 1,
    dpi: int = 100,
) -> Path:
    """Render an MP4 of the footage with a synced, scrolling motion-energy plot.

    The full grayscale luma frame is shown on top, with each ROI outlined, and
    the selected motion-energy traces scroll below it with a thin cursor. The
    displayed pixels are the exact Y-plane array motion energy was computed on
    (identical to native for monochrome cameras).

    `trace` is frame-aligned: one row per rendered frame, ``trace[0]`` being
    the first frame (see `align_to_frames`); NaNs (such as frame 0) show as
    gaps in the plotted line. The x-axis is source time in
    seconds, independent of `out_fps`. `out_fps` only sets playback speed of
    the output file (no frames are dropped).

    Parameters
    ----------
    video_path : Path
        Path to the source video.
    trace : numpy.ndarray
        Cleaned (interpolated) motion energy, shape ``(T,)`` or ``(T, 1 + K)``
        with the full-frame trace in column 0.
    fps_source : float
        Frame rate of the source video, used to convert frame index to time.
    output_path : Path
        Path to write the rendered MP4 to.
    raw_trace : numpy.ndarray or None, optional
        Unmodified motion energy (same shape as `trace`) including keyframe
        artifact spikes. When given, the raw full-frame trace is drawn as a
        faint gray background line under the cleaned traces, so a viewer can
        see encoding spikes at their true amplitude and position.
    roi_ids : Sequence[str] or None, optional
        Name of each column of `trace`; defaults to ``full_frame, roi1, ...``.
    rois : dict[str, numpy.ndarray] or None, optional
        ROIs to outline on the frame, coloured to match their traces.
    viz_rois : Sequence[str] or None, optional
        Ids of the traces to plot. By default the first five columns
        (full_frame plus the first four ROIs).
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
        If `stride` is less than 1, or `viz_rois` names an unknown id.
    """
    plt = _import_pyplot()

    output_path = Path(output_path)
    clean = _as_columns(trace)
    ids = _column_ids(clean.shape[1], roi_ids)
    raw = _as_columns(raw_trace)[:, 0] if raw_trace is not None else None
    start = start_frame or 0
    if stride < 1:
        raise ValueError(f"stride must be >= 1 (got {stride})")
    traces = _select_traces(plt, clean, ids, viz_rois)
    if not any(label == FULL_FRAME_ID for label, _, _ in traces):
        raw = None

    # trace[m] belongs to frame (start + m); place it at that frame's time.
    trace_t = (start + np.arange(len(clean))) / fps_source
    fig, ax_img, ax_plot, cursor = _build_trace_figure(plt, dpi, trace_t, traces, raw)

    img_artist = None
    container: Any = None
    stream: Any = None
    try:
        for j, (frame, _is_key) in enumerate(
            iter_luma_frames(video_path, start_frame=start_frame, end_frame=end_frame)
        ):
            if j % stride != 0:
                continue
            t_now = (start + j) / fps_source

            if img_artist is None:
                img_artist = ax_img.imshow(frame, cmap="gray", vmin=0, vmax=255, aspect="equal")
                # Outlines go on after the image so they share its pixel axes.
                if rois is not None:
                    _draw_roi_outlines(plt, ax_img, rois)
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
