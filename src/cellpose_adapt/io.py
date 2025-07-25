import logging
import os
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
# from cellpose import io
from skimage import io

logger = logging.getLogger(__name__)
logger.debug("Image I/O module loaded. Using skimage.io for image reading.")

@lru_cache(maxsize=64)
def _read_image_from_disk(path: str) -> Optional[np.ndarray]:
    """A cached function to read an image file from disk."""
    if not path or not os.path.exists(path):
        logger.warning("File not found for caching: %s", path)
        return None
    logger.debug("CACHE MISS: Reading file from disk: %s", path)
    return io.imread(path)


def load_image_with_gt(
        image_path: str,
        ground_truth_path: str,
        channel_to_segment: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads an image and GT mask, using a cache for disk reads.
    If channel_to_segment is specified, it extracts that channel for segmentation.
    """
    # Use the cached function for the slow I/O operations
    original_image = _read_image_from_disk(image_path)
    ground_truth = _read_image_from_disk(ground_truth_path) if ground_truth_path else None

    if ground_truth_path and ground_truth is None:
        logging.error(f"Ground truth was not found at the inferred path: {ground_truth_path}")

    if original_image is None:
        logging.error(f"Failed to load image from {image_path}")
        return None, ground_truth, None

    # --- Fast Processing: Channel Selection ---
    # This part is fast and doesn't need to be cached itself.
    image_for_segmentation = original_image
    if channel_to_segment is not None:
        # Assuming shape (Z, C, Y, X) for 3D or (C, Y, X) for 2D
        # Determine which axis is the channel axis
        channel_axis = -1
        if original_image.ndim == 4:  # Z, C, Y, X
            channel_axis = 1
        elif original_image.ndim == 3 and original_image.shape[0] < 5:  # C, Y, X (heuristic)
            channel_axis = 0

        if channel_axis != -1 and original_image.shape[channel_axis] > channel_to_segment:
            logger.debug(f"Selecting channel {channel_to_segment} from image with shape {original_image.shape}.")
            # Use np.take to select the slice along the correct axis
            image_for_segmentation = np.take(original_image, indices=channel_to_segment, axis=channel_axis)
        elif original_image.ndim > 2:
            logger.warning(
                f"Cannot select channel {channel_to_segment}. Image has shape {original_image.shape}. Using as-is."
            )

    return image_for_segmentation, ground_truth, original_image


def find_gt_path(image_path: str, mapping_rules: dict) -> str:
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
    img_suffix = gt_mapping.get("img_suffix", ".tif") if gt_mapping is not None else ""

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

                    gt_path = find_gt_path(image_path, gt_mapping) if gt_mapping else None

                    if gt_mapping and os.path.exists(gt_path):
                        source_pairs.append((image_path, gt_path))
                        logger.debug("Found pair: IMG='%s', GT='%s'", image_path, gt_path)
                    elif gt_mapping:
                        logger.debug("Image '%s' found, but corresponding GT '%s' does not exist.",image_path,gt_path)
                    else:
                        source_pairs.append((image_path, None))
                        logger.debug("Image '%s' found without GT mapping rules.", image_path)

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
