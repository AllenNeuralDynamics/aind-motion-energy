"""Regions of interest as named binary masks.

Masks are the common pathway for every ROI shape: rectangles, polygons and
compound shapes are all rasterized by the ROI manager (the annotator app) into
one boolean mask per ROI. Here a set of ROIs is a plain ``dict`` mapping each
ROI name to its ``(H, W)`` mask; insertion order is column order in the output.
"""

from pathlib import Path

import numpy as np

FULL_FRAME_ID = "full_frame"


def load_roi_masks(path: Path) -> dict[str, np.ndarray]:
    """Load an ROI annotator ``*_masks.npz`` file.

    Parameters
    ----------
    path : Path
        NPZ file with ``masks`` (``(H, W, K)``, 0/1) and ``roi_ids`` (``(K,)``).

    Returns
    -------
    dict[str, numpy.ndarray]
        ROI name -> bool mask of shape ``(H, W)``, in file order.

    Raises
    ------
    ValueError
        If the file lacks either array, the masks aren't 3-D and binary, or
        the ids don't match the mask channels or aren't unique.
    """
    with np.load(path, allow_pickle=False) as f:
        missing = {"masks", "roi_ids"} - set(f.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}; expected masks and roi_ids")
        masks, ids = f["masks"], [str(i) for i in f["roi_ids"]]
    if masks.ndim != 3:
        raise ValueError(f"{path}: masks must have shape (H, W, K); got {masks.shape}")
    if len(ids) != masks.shape[2]:
        raise ValueError(f"{path}: {len(ids)} roi_ids for {masks.shape[2]} mask channels")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: roi_ids must be unique; got {ids}")
    if masks.dtype != bool and not np.isin(masks, (0, 1)).all():
        raise ValueError(f"{path}: masks must be binary (only 0 and 1)")
    return {roi_id: masks[:, :, k].astype(bool) for k, roi_id in enumerate(ids)}


def box_roi(box: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """One rectangular ROI named ``"roi"``, for ``--roi X Y W H``.

    Parameters
    ----------
    box : tuple[int, int, int, int]
        ``(x, y, w, h)`` in pixels.
    frame_shape : tuple[int, int]
        ``(H, W)`` of the video frame.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``{"roi": mask}``.

    Raises
    ------
    ValueError
        If the box is empty or extends outside the frame.
    """
    x, y, w, h = box
    frame_h, frame_w = frame_shape
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise ValueError(f"ROI box {box} does not fit inside the {frame_w}x{frame_h} frame")
    mask = np.zeros(frame_shape, dtype=bool)
    mask[y : y + h, x : x + w] = True
    return {"roi": mask}


def check_rois(rois: dict[str, np.ndarray], height: int, width: int) -> None:
    """Raise unless every ROI is a non-empty mask of a ``height`` x ``width`` frame.

    Parameters
    ----------
    rois : dict[str, numpy.ndarray]
        ROI name -> mask.
    height : int
        Video frame height in pixels.
    width : int
        Video frame width in pixels.

    Raises
    ------
    ValueError
        If a mask has the wrong shape or no pixels, or an ROI uses the
        reserved name ``"full_frame"``.
    """
    if FULL_FRAME_ID in rois:
        raise ValueError(f"{FULL_FRAME_ID!r} is reserved for the whole-frame trace")
    for roi_id, mask in rois.items():
        if np.shape(mask) != (height, width):
            shape = "x".join(map(str, np.shape(mask)[::-1]))
            raise ValueError(f"ROI {roi_id!r} is {shape} but the video is {width}x{height}")
        if not np.any(mask):
            raise ValueError(f"ROI {roi_id!r} has an empty mask")
