import logging
import os
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from cellpose import io

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def load_image_with_gt(
    image_path: str, ground_truth_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image and its corresponding ground truth mask.
    Results are cached in memory to avoid repeated disk reads.
    """

    logger.debug("Loading image from: %s", image_path)
    image, ground_truth = None, None
    if os.path.exists(image_path):
        image = io.imread(image_path)
    else:
        logger.warning("Image not found: %s. Skipping.", image_path)

    if ground_truth_path and os.path.exists(ground_truth_path):
        ground_truth = io.imread(ground_truth_path)
    elif ground_truth_path:
        logger.warning("Ground truth not found: %s. Skipping.", ground_truth_path)

    # For 3D images, ensure that the channel axis is the last dimension
    if image is not None and image.ndim == 3:
        if (image.ndim == 3) and (image.shape[0] == 3):
            image = image.transpose((1, 2, 0))
    if ground_truth is not None and ground_truth.ndim == 3:
        if (ground_truth.ndim == 3) and (ground_truth.shape[0] == 3):
            ground_truth = ground_truth.transpose((1, 2, 0))

    return image, ground_truth


def _find_gt_path(image_path: str, mapping_rules: dict) -> str:
    """
    Constructs a ground truth path from an image path based on mapping rules.
    """
    # ... (function content is unchanged) ...
    gt_path = image_path

    replacements = mapping_rules.get("replace", [])
    for old, new in replacements:
        gt_path = gt_path.replace(old, new)

    img_suffix = mapping_rules.get("img_suffix", ".tif")
    gt_suffix = mapping_rules.get("suffix", "_masks.tif")

    base_name = os.path.basename(gt_path)
    if base_name.endswith(img_suffix):
        base_name = base_name[: -len(img_suffix)]

    new_filename = base_name + gt_suffix
    gt_path = os.path.join(os.path.dirname(gt_path), new_filename)

    return gt_path


def find_image_gt_pairs(
    data_dirs: List[str], gt_mapping: dict, limit_per_source: int = None
) -> List[Tuple[str, str]]:
    """
    Scans multiple directories for images and constructs ground truth paths using flexible mapping rules.
    Can limit the number of images taken from each data source.

    Args:
        data_dirs (List[str]): A list of directories to search for images.
        gt_mapping (dict): A dictionary defining how to map an image path to a ground truth path.
        limit_per_source (int, optional): The maximum number of images to take from each directory.
                                          If None, all images are taken. Defaults to None.

    Returns:
        List[Tuple[str, str]]: A list of (image_path, ground_truth_path) pairs.
    """
    all_pairs = []
    img_suffix = gt_mapping.get("img_suffix", ".tif")

    if limit_per_source:
        logger.info(
            "Scanning for images with suffix '%s' in directories: %s (limit %d per source)",
            img_suffix,
            data_dirs,
            limit_per_source,
        )
    else:
        logger.info(
            "Scanning for images with suffix '%s' in directories: %s (no limit)",
            img_suffix,
            data_dirs,
        )

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            logger.warning("Data directory not found, skipping: %s", data_dir)
            continue

        source_pairs = []
        # Sort files to ensure consistent selection when limiting
        for root, _, files in sorted(os.walk(data_dir)):
            for file in sorted(files):
                # Check if we have already reached the limit for this source
                if limit_per_source and len(source_pairs) >= limit_per_source:
                    break

                if file.endswith(img_suffix):
                    image_path = os.path.join(root, file)
                    gt_path = _find_gt_path(image_path, gt_mapping)

                    if os.path.exists(gt_path):
                        source_pairs.append((image_path, gt_path))
                        logger.debug(
                            "Found pair: IMG='%s', GT='%s'", image_path, gt_path
                        )
                    else:
                        logger.debug(
                            "Image '%s' found, but corresponding GT '%s' does not exist.",
                            image_path,
                            gt_path,
                        )

            # Break from inner loops if the limit is reached
            if limit_per_source and len(source_pairs) >= limit_per_source:
                break

        logger.info("Found %d pairs in source: %s", len(source_pairs), data_dir)
        all_pairs.extend(source_pairs)

    logger.info(
        "Total image-ground truth pairs found across all sources: %d", len(all_pairs)
    )
    if not all_pairs:
        logger.warning(
            "No data pairs found. Check your `data_sources` and `gt_mapping` rules in the project config."
        )

    return all_pairs
