import numpy as np
import pytest

from aind_motion_energy.rois import box_roi, check_rois, load_roi_masks


def test_load_roi_masks_reads_annotator_export(tmp_path):
    # Same layout export_rois.py writes: compressed, uint8 masks, unicode ids.
    masks = np.zeros((5, 6, 2), np.uint8)
    masks[1:3, 1:4, 0] = 1
    masks[4, 5, 1] = 1
    path = tmp_path / "sess_Cam_masks.npz"
    np.savez_compressed(path, masks=masks, roi_ids=np.array(["Cam_Snout_box", "Cam_Pupil_polygon"]))

    rois = load_roi_masks(path)

    assert list(rois) == ["Cam_Snout_box", "Cam_Pupil_polygon"]  # file order kept
    assert all(m.dtype == bool and m.shape == (5, 6) for m in rois.values())
    assert [int(m.sum()) for m in rois.values()] == [6, 1]


def test_load_roi_masks_accepts_bool_masks(tmp_path):
    path = tmp_path / "x_masks.npz"
    np.savez(path, masks=np.ones((2, 2, 1), bool), roi_ids=np.array(["a"]))
    assert load_roi_masks(path)["a"].all()


@pytest.mark.parametrize(
    "masks,ids,match",
    [
        (np.ones((2, 2), np.uint8), ["a"], "shape"),
        (np.ones((2, 2, 2), np.uint8), ["a"], "1 roi_ids for 2"),
        (np.ones((2, 2, 2), np.uint8), ["a", "a"], "unique"),
        (np.full((2, 2, 1), 2, np.uint8), ["a"], "binary"),
    ],
)
def test_load_roi_masks_validation(tmp_path, masks, ids, match):
    path = tmp_path / "bad_masks.npz"
    np.savez(path, masks=masks, roi_ids=np.array(ids))
    with pytest.raises(ValueError, match=match):
        load_roi_masks(path)


def test_load_roi_masks_missing_arrays(tmp_path):
    path = tmp_path / "bad_masks.npz"
    np.savez(path, masks=np.ones((2, 2, 1), np.uint8))
    with pytest.raises(ValueError, match="roi_ids"):
        load_roi_masks(path)


def test_box_roi():
    rois = box_roi((2, 1, 4, 3), (6, 8))
    assert list(rois) == ["roi"]
    expected = np.zeros((6, 8), bool)
    expected[1:4, 2:6] = True
    np.testing.assert_array_equal(rois["roi"], expected)


@pytest.mark.parametrize("box", [(0, 0, 0, 2), (-1, 0, 2, 2), (6, 0, 3, 2), (0, 5, 2, 2)])
def test_box_roi_rejects_empty_or_out_of_frame(box):
    with pytest.raises(ValueError, match="does not fit"):
        box_roi(box, (6, 8))


def test_check_rois():
    check_rois({"a": np.ones((3, 4), bool)}, 3, 4)
    with pytest.raises(ValueError, match="ROI 'a' is 4x3 but the video is 5x3"):
        check_rois({"a": np.ones((3, 4), bool)}, 3, 5)
    with pytest.raises(ValueError, match="empty"):
        check_rois({"a": np.zeros((3, 4), bool)}, 3, 4)
    with pytest.raises(ValueError, match="reserved"):
        check_rois({"full_frame": np.ones((3, 4), bool)}, 3, 4)
