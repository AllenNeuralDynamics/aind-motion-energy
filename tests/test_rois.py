import numpy as np
import pytest

from aind_motion_energy import rois as rois_mod
from aind_motion_energy.rois import RoiSet, load_roi_masks


def _disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r**2


def test_reduce_matches_brute_force_for_overlapping_and_compound_masks():
    H, W = 40, 50
    outer = _disk(H, W, 20, 25, 15)
    inner = _disk(H, W, 20, 25, 6)
    masks = np.stack(
        [
            outer & ~inner,  # ring: a polygon with a hole
            inner,  # overlaps the ring's hole edge region
            np.pad(np.ones((5, 5), bool), ((0, H - 5), (0, W - 5))),  # corner box
            np.eye(H, W, dtype=bool),  # thin diagonal line
            np.zeros((H, W), bool) | (np.arange(W) == 7)[None, :],  # single column
        ],
        axis=-1,
    ).astype(np.uint8)
    rois = RoiSet.from_masks(masks, ["ring", "disk", "box", "diag", "col"])
    rng = np.random.default_rng(0)
    diff = rng.integers(0, 256, (H, W)).astype(np.uint16)

    sums = rois.reduce(diff)

    expected = [int(diff[masks[..., k] == 1].sum()) for k in range(masks.shape[2])]
    np.testing.assert_array_equal(sums, expected)
    assert sums.dtype == np.uint32


def test_single_pixel_roi():
    masks = np.zeros((4, 4, 1), np.uint8)
    masks[2, 3, 0] = 1
    rois = RoiSet.from_masks(masks, ["px"])
    diff = np.arange(16, dtype=np.uint16).reshape(4, 4)
    assert rois.reduce(diff).tolist() == [diff[2, 3]]
    np.testing.assert_array_equal(rois.bboxes[0], [3, 2, 1, 1])
    assert rois.pixel_counts.tolist() == [1]


def test_from_box_matches_slice_sum_and_metadata():
    rois = RoiSet.from_box((2, 1, 4, 3), (6, 8))
    diff = np.arange(48, dtype=np.uint16).reshape(6, 8)
    assert rois.reduce(diff).tolist() == [int(diff[1:4, 2:6].sum())]
    assert rois.ids == ("roi",)
    assert len(rois) == 1
    np.testing.assert_array_equal(rois.bboxes[0], [2, 1, 4, 3])
    assert rois.frame_shape == (6, 8)


@pytest.mark.parametrize("box", [(0, 0, 0, 2), (-1, 0, 2, 2), (6, 0, 3, 2), (0, 5, 2, 2)])
def test_from_box_rejects_empty_or_out_of_frame(box):
    with pytest.raises(ValueError, match="does not fit"):
        RoiSet.from_box(box, (6, 8))


def test_mask_round_trips_cropped_mask():
    H, W = 30, 30
    m = _disk(H, W, 12, 18, 7)
    rois = RoiSet.from_masks(m[..., None], ["d"])
    (x, y, w, h), cropped = rois.mask(0)
    rebuilt = np.zeros((H, W), bool)
    rebuilt[y : y + h, x : x + w] = cropped
    np.testing.assert_array_equal(rebuilt, m)


def test_bool_masks_accepted():
    rois = RoiSet.from_masks(np.ones((2, 2, 1), bool), ["a"])
    assert rois.pixel_counts.tolist() == [4]


@pytest.mark.parametrize(
    "masks,ids,match",
    [
        (np.ones((2, 2), np.uint8), ["a"], "shape"),
        (np.ones((2, 2, 0), np.uint8), [], "no ROIs"),
        (np.ones((2, 2, 2), np.uint8), ["a"], "1 ROI ids for 2"),
        (np.ones((2, 2, 2), np.uint8), ["a", "a"], "unique"),
        (np.ones((2, 2, 1), np.uint8), ["full_frame"], "reserved"),
        (np.full((2, 2, 1), 2, np.uint8), ["a"], "binary"),
        (np.zeros((2, 2, 1), np.uint8), ["a"], "empty"),
    ],
)
def test_from_masks_validation(masks, ids, match):
    with pytest.raises(ValueError, match=match):
        RoiSet.from_masks(masks, ids)


def test_check_frame_shape():
    rois = RoiSet.from_box((0, 0, 1, 1), (3, 4))
    rois.check_frame_shape(3, 4)
    with pytest.raises(ValueError, match="4x3 but the video is 5x3"):
        rois.check_frame_shape(3, 5)


def test_total_area_warns_then_errors(monkeypatch):
    monkeypatch.setattr(rois_mod, "AREA_WARN_FACTOR", 1.5)
    monkeypatch.setattr(rois_mod, "AREA_ERROR_FACTOR", 2.5)
    full = np.ones((3, 3, 2), np.uint8)  # 2x the frame area
    with pytest.warns(UserWarning, match="2.0x the frame area"):
        RoiSet.from_masks(full, ["a", "b"])
    with pytest.raises(ValueError, match="Limit is 2.5x"):
        RoiSet.from_masks(np.ones((3, 3, 3), np.uint8), ["a", "b", "c"])


def test_many_rois_warn(monkeypatch):
    monkeypatch.setattr(rois_mod, "COUNT_WARN", 2)
    masks = np.zeros((4, 4, 3), np.uint8)
    for k in range(3):
        masks[k, k, k] = 1
    with pytest.warns(UserWarning, match="3 ROIs"):
        RoiSet.from_masks(masks, ["a", "b", "c"])


def test_load_roi_masks_reads_annotator_export(tmp_path):
    # Same layout export_rois.py writes: compressed, uint8 masks, unicode ids.
    masks = np.zeros((5, 6, 2), np.uint8)
    masks[1:3, 1:4, 0] = 1
    masks[4, 5, 1] = 1
    path = tmp_path / "sess_Cam_masks.npz"
    np.savez_compressed(path, masks=masks, roi_ids=np.array(["Cam_Snout_box", "Cam_Pupil_polygon"]))

    rois = load_roi_masks(path)

    assert rois.ids == ("Cam_Snout_box", "Cam_Pupil_polygon")
    assert rois.frame_shape == (5, 6)
    assert rois.pixel_counts.tolist() == [6, 1]


def test_load_roi_masks_missing_arrays(tmp_path):
    path = tmp_path / "bad_masks.npz"
    np.savez(path, masks=np.ones((2, 2, 1), np.uint8))
    with pytest.raises(ValueError, match="roi_ids"):
        load_roi_masks(path)


def test_accumulator_widens_only_when_uint32_could_overflow():
    limit = np.iinfo(np.uint32).max // 255
    assert rois_mod._sum_dtype(limit) == np.uint32
    assert rois_mod._sum_dtype(limit + 1) == np.uint64
