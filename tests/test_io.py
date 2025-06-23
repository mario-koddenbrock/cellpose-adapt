import os

from cellpose_adapt.io import find_gt_path, find_image_gt_pairs


# === Unit tests for find_gt_path (no file system needed) ===

def test_find_gt_path_simple_suffix():
    """Tests a simple suffix change."""
    image_path = "data/folder/image_01.tif"
    mapping_rules = {"suffix": "_masks.tif", "img_suffix": ".tif"}
    expected_gt_path = os.path.join("data", "folder", "image_01_masks.tif")
    assert find_gt_path(image_path, mapping_rules) == expected_gt_path

def test_find_gt_path_with_replace():
    """Tests directory part replacement."""
    image_path = os.path.join("data", "images", "set_A", "image_01.tif")
    mapping_rules = {
        "replace": [["images", "masks"]],
        "suffix": ".tif",
        "img_suffix": ".tif"
    }
    expected_gt_path = os.path.join("data", "masks", "set_A", "image_01.tif")
    assert find_gt_path(image_path, mapping_rules) == expected_gt_path

def test_find_gt_path_complex():
    """Tests a combination of replacement and suffix change."""
    image_path = os.path.join("raw_data", "exp1", "images", "cell.tif")
    mapping_rules = {
        "replace": [["images", os.path.join("labels", "final")]],
        "suffix": "_GT.tif",
        "img_suffix": ".tif"
    }
    expected_gt_path = os.path.join("raw_data", "exp1", "labels", "final", "cell_GT.tif")
    assert find_gt_path(image_path, mapping_rules) == expected_gt_path


# === Integration tests for find_image_gt_pairs (uses tmp_path fixture) ===

def test_find_image_gt_pairs_single_source(tmp_path):
    """Tests finding pairs in a single directory structure."""
    # 1. Create a mock file system in the temporary directory
    data_dir = tmp_path / "dataset1"
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir()

    # Create dummy files
    (img_dir / "img1.tif").touch()
    (mask_dir / "img1_masks.tif").touch()
    (img_dir / "img2.tif").touch()
    # No mask for img2, should be ignored
    (img_dir / "img3.tif").touch()
    (mask_dir / "img3_masks.tif").touch()
    (img_dir / "not_an_image.txt").touch() # Should be ignored

    # 2. Define mapping rules and call the function
    gt_mapping = {"replace": [["images", "masks"]], "suffix": "_masks.tif", "img_suffix": ".tif"}
    data_sources = [str(img_dir)]

    pairs = find_image_gt_pairs(data_sources, gt_mapping)

    # 3. Assert the results
    assert len(pairs) == 2
    # Convert paths to sets for order-independent comparison
    found_imgs = {os.path.basename(p[0]) for p in pairs}
    assert found_imgs == {"img1.tif", "img3.tif"}

    # Check one pair explicitly
    expected_pair_1 = (str(img_dir / "img1.tif"), str(mask_dir / "img1_masks.tif"))
    assert expected_pair_1 in pairs


def test_find_image_gt_pairs_multiple_sources(tmp_path):
    """Tests finding pairs across multiple data source directories."""
    # 1. Create mock file systems
    # Source A
    source_a = tmp_path / "source_A"
    (source_a / "images").mkdir(parents=True)
    (source_a / "labels").mkdir()
    (source_a / "images" / "a1.png").touch()
    (source_a / "labels" / "a1_gt.png").touch()

    # Source B
    source_b = tmp_path / "source_B"
    (source_b / "raw").mkdir(parents=True)
    (source_b / "gt").mkdir()
    (source_b / "raw" / "b1.png").touch()
    (source_b / "gt" / "b1_gt.png").touch()
    (source_b / "raw" / "b2.png").touch() # No mask

    # 2. Define mapping rules and call the function
    gt_mapping = {"replace": [["images", "labels"], ["raw", "gt"]], "suffix": "_gt.png", "img_suffix": ".png"}
    data_sources = [str(source_a / "images"), str(source_b / "raw")]

    pairs = find_image_gt_pairs(data_sources, gt_mapping)

    # 3. Assert the results
    assert len(pairs) == 2

    expected_pair_a = (str(source_a / "images" / "a1.png"), str(source_a / "labels" / "a1_gt.png"))
    expected_pair_b = (str(source_b / "raw" / "b1.png"), str(source_b / "gt" / "b1_gt.png"))

    assert expected_pair_a in pairs
    assert expected_pair_b in pairs