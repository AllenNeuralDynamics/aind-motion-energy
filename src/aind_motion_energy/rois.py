"""Regions of interest as binary masks, reduced in a single pass per frame.

Masks are the common pathway for every ROI shape: rectangles, polygons and
compound shapes are all rasterized by the ROI manager (the annotator app) into
one boolean mask per ROI, and this module only consumes those masks.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FULL_FRAME_ID = "full_frame"

# Size guards. Per-frame cost and index memory scale with the total number of
# masked pixels, not with the number of ROIs, so the hard limit is on area.
AREA_WARN_FACTOR = 2.0
AREA_ERROR_FACTOR = 20.0
COUNT_WARN = 256


@dataclass(frozen=True, eq=False)
class RoiSet:
    """A set of named binary-mask ROIs on one camera's frame.

    Stored compactly as the concatenated flat pixel indices of every mask
    rather than as dense masks, so many full-resolution masks don't stay
    resident. Build with `from_masks`, `from_box` or `load_roi_masks`.

    Attributes
    ----------
    ids : tuple[str, ...]
        ROI names, in channel order.
    frame_shape : tuple[int, int]
        ``(H, W)`` of the frame the masks were drawn on.
    bboxes : numpy.ndarray
        Int32 array of shape ``(K, 4)``, each ROI's ``(x, y, w, h)`` bounding box.
    pixel_counts : numpy.ndarray
        Int64 array of shape ``(K,)``, the number of pixels in each mask.
    flat_idx : numpy.ndarray
        Flat (row-major) pixel indices of every mask, concatenated in ROI order.
        Stored as ``intp``: NumPy fancy indexing with native-width indices is
        ~4x faster than `numpy.take` or narrower index types.
    offsets : numpy.ndarray
        Start of each ROI's segment within `flat_idx`.
    sum_dtype : numpy.dtype
        Accumulator for `reduce`: ``uint32`` (fastest) whenever no mask is large
        enough for an 8-bit diff sum to overflow it, else ``uint64``.
    """

    ids: tuple[str, ...]
    frame_shape: tuple[int, int]
    bboxes: np.ndarray
    pixel_counts: np.ndarray
    flat_idx: np.ndarray
    offsets: np.ndarray
    sum_dtype: np.dtype[Any]

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_masks(cls, masks: np.ndarray, roi_ids: list[str] | tuple[str, ...]) -> "RoiSet":
        """Build a `RoiSet` from stacked binary masks.

        Parameters
        ----------
        masks : numpy.ndarray
            Array of shape ``(H, W, K)`` containing only 0/1 (or bool) values,
            as written by the ROI annotator's ``*_masks.npz`` export.
        roi_ids : list[str] or tuple[str, ...]
            Name of each mask channel, length ``K``.

        Returns
        -------
        RoiSet
            The validated ROI set.

        Raises
        ------
        ValueError
            If the masks aren't 3-D and binary, if the ids don't match the
            channels or aren't unique, if an id is the reserved
            ``"full_frame"``, if a mask is empty, or if the masks together
            cover more than ``AREA_ERROR_FACTOR`` times the frame area.
        """
        masks = np.asarray(masks)
        ids = tuple(str(i) for i in roi_ids)
        _validate_masks(masks, ids)

        h, w, _ = masks.shape
        bboxes, counts, segments = [], [], []
        for k, roi_id in enumerate(ids):
            plane = masks[:, :, k].astype(bool)
            idx = np.flatnonzero(plane)
            if idx.size == 0:
                raise ValueError(f"ROI {roi_id!r} has an empty mask")
            rows, cols = np.divmod(idx, w)
            x0, y0 = int(cols.min()), int(rows.min())
            bboxes.append((x0, y0, int(cols.max()) - x0 + 1, int(rows.max()) - y0 + 1))
            counts.append(idx.size)
            segments.append(idx)

        pixel_counts = np.array(counts, dtype=np.int64)
        _check_total_area(pixel_counts, h * w, len(ids))
        return cls(
            ids=ids,
            frame_shape=(h, w),
            bboxes=np.array(bboxes, dtype=np.int32).reshape(-1, 4),
            pixel_counts=pixel_counts,
            flat_idx=np.concatenate(segments).astype(np.intp),
            offsets=np.concatenate([[0], np.cumsum(pixel_counts)[:-1]]).astype(np.intp),
            sum_dtype=_sum_dtype(int(pixel_counts.max())),
        )

    @classmethod
    def from_box(
        cls,
        box: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
        roi_id: str = "roi",
    ) -> "RoiSet":
        """Build a single rectangular ROI.

        Parameters
        ----------
        box : tuple[int, int, int, int]
            ``(x, y, w, h)`` in pixels.
        frame_shape : tuple[int, int]
            ``(H, W)`` of the video frame.
        roi_id : str, optional
            Name of the ROI, by default ``"roi"``.

        Returns
        -------
        RoiSet
            A set containing one rectangular mask.

        Raises
        ------
        ValueError
            If the box is empty or extends outside the frame.
        """
        x, y, w, h = box
        frame_h, frame_w = frame_shape
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            raise ValueError(f"ROI box {box} does not fit inside the {frame_w}x{frame_h} frame")
        mask = np.zeros((frame_h, frame_w, 1), dtype=bool)
        mask[y : y + h, x : x + w, 0] = True
        return cls.from_masks(mask, [roi_id])

    def check_frame_shape(self, height: int, width: int) -> None:
        """Raise if these masks weren't drawn on a ``height`` x ``width`` frame.

        Parameters
        ----------
        height : int
            Video frame height in pixels.
        width : int
            Video frame width in pixels.

        Raises
        ------
        ValueError
            If the shapes differ.
        """
        if self.frame_shape != (height, width):
            mh, mw = self.frame_shape
            raise ValueError(f"ROI masks are {mw}x{mh} but the video is {width}x{height}")

    def reduce(self, diff: np.ndarray) -> np.ndarray:
        """Sum `diff` inside every mask in one vectorized pass.

        Parameters
        ----------
        diff : numpy.ndarray
            Absolute 8-bit frame difference (values 0-255) of shape
            `frame_shape`.

        Returns
        -------
        numpy.ndarray
            Unsigned integer array (`sum_dtype`) of shape ``(K,)``, the exact
            per-ROI sums.
        """
        vals = diff.ravel()[self.flat_idx]
        return np.add.reduceat(vals, self.offsets, dtype=self.sum_dtype)

    def mask(self, i: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
        """Return ROI `i`'s bounding box and its mask cropped to that box.

        Parameters
        ----------
        i : int
            ROI index.

        Returns
        -------
        bbox : tuple[int, int, int, int]
            ``(x, y, w, h)`` of the ROI.
        cropped : numpy.ndarray
            Bool array of shape ``(h, w)``.
        """
        x, y, w, h = (int(v) for v in self.bboxes[i])
        start = int(self.offsets[i])
        idx = self.flat_idx[start : start + int(self.pixel_counts[i])]
        rows, cols = np.divmod(idx, self.frame_shape[1])
        cropped = np.zeros((h, w), dtype=bool)
        cropped[rows - y, cols - x] = True
        return (x, y, w, h), cropped


def _sum_dtype(max_pixels: int) -> np.dtype[Any]:
    """Narrowest exact accumulator for a mask of `max_pixels` (8-bit diffs: <= 255 each)."""
    fits_u32 = max_pixels * 255 <= np.iinfo(np.uint32).max
    return np.dtype(np.uint32 if fits_u32 else np.uint64)


def _validate_masks(masks: np.ndarray, ids: tuple[str, ...]) -> None:
    if masks.ndim != 3:
        raise ValueError(f"ROI masks must have shape (H, W, K); got {masks.shape}")
    if masks.shape[2] == 0:
        raise ValueError("ROI masks contain no ROIs")
    if len(ids) != masks.shape[2]:
        raise ValueError(f"{len(ids)} ROI ids for {masks.shape[2]} mask channels")
    if len(set(ids)) != len(ids):
        raise ValueError(f"ROI ids must be unique; got {list(ids)}")
    if FULL_FRAME_ID in ids:
        raise ValueError(f"{FULL_FRAME_ID!r} is reserved for the whole-frame trace")
    if masks.dtype != bool and not np.isin(masks, (0, 1)).all():
        raise ValueError("ROI masks must be binary (only 0 and 1)")


def _check_total_area(pixel_counts: np.ndarray, frame_area: int, n_rois: int) -> None:
    total = int(pixel_counts.sum())
    factor = total / frame_area
    detail = (
        f"ROI masks cover {total:,} pixels in total ({factor:.1f}x the frame area); "
        f"per-frame cost and index memory (~{total * 8 / 1e6:.0f} MB) scale with this"
    )
    if factor > AREA_ERROR_FACTOR:
        raise ValueError(f"{detail}. Limit is {AREA_ERROR_FACTOR:g}x — check the mask export.")
    if factor > AREA_WARN_FACTOR:
        warnings.warn(detail, stacklevel=3)
    if n_rois > COUNT_WARN:
        warnings.warn(
            f"{n_rois} ROIs: outputs grow with the ROI count and summary plots get hard to read",
            stacklevel=3,
        )


def load_roi_masks(path: Path) -> RoiSet:
    """Load an ROI annotator ``*_masks.npz`` file.

    Parameters
    ----------
    path : Path
        NPZ file with ``masks`` (``(H, W, K)`` 0/1) and ``roi_ids`` (``(K,)``).

    Returns
    -------
    RoiSet
        The validated ROI set.

    Raises
    ------
    ValueError
        If the file lacks either array, or the masks fail validation (see
        `RoiSet.from_masks`).
    """
    with np.load(path, allow_pickle=False) as f:
        missing = {"masks", "roi_ids"} - set(f.files)
        if missing:
            raise ValueError(f"{path} is missing {sorted(missing)}; expected masks and roi_ids")
        return RoiSet.from_masks(f["masks"], [str(i) for i in f["roi_ids"]])
