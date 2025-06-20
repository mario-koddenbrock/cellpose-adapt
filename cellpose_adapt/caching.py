import hashlib
import json
import logging
import os

import numpy as np

from cellpose_adapt.config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)
CACHE_DIR = ".cache"
# Ensure the cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
logger.debug("Using cache directory: %s", CACHE_DIR)


def get_model_eval_params(config: PipelineConfig) -> dict:
    """
    Filters the config to get only parameters that affect the raw model.eval output.
    This is crucial for caching, as it ignores post-processing parameters.
    """
    model_keys = [
        "model_name",
        "diameter",
        "invert",
        "normalize",
        "norm3D",
        "percentile_min",
        "percentile_max",
        "sharpen_radius",
        "smooth_radius",
        "do_3D",
        "tile_overlap",
        "stitch_threshold",
        "channel_axis",
        "z_axis",
    ]
    return {k: getattr(config, k) for k in model_keys}


def compute_hash(image: np.ndarray, params: dict) -> str:
    """Computes a unique SHA256 hash for an image and a dictionary of parameters."""
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    param_string = json.dumps(params, sort_keys=True)
    param_hash = hashlib.sha256(param_string.encode()).hexdigest()
    return f"{image_hash}_{param_hash}"


def save_to_cache(cache_key: str, flows: list, styles: np.ndarray):
    """Saves the raw output of model.eval to the on-disk cache."""
    cache_key_dir = os.path.join(CACHE_DIR, cache_key)
    os.makedirs(cache_key_dir, exist_ok=True)

    try:
        np.save(os.path.join(cache_key_dir, "styles.npy"), styles.astype(np.float32))

        # Save flows as separate files
        flows_dir = os.path.join(cache_key_dir, "flows")
        os.makedirs(flows_dir, exist_ok=True)
        # dP_colors, dP, cellprob, p_interp
        np.save(os.path.join(flows_dir, "0_dP_colors.npy"), flows[0].astype(np.uint8))
        np.save(os.path.join(flows_dir, "1_dP.npy"), flows[1].astype(np.float32))
        np.save(os.path.join(flows_dir, "2_cellprob.npy"), flows[2].astype(np.float32))
        if len(flows) > 3:
            np.save(
                os.path.join(flows_dir, "3_p_interp.npy"), flows[3].astype(np.float32)
            )

        logger.debug("Saved prediction to cache with key %s", cache_key)
    except Exception as e:
        logger.error("Failed to save to cache: %s", e)


def load_from_cache(cache_key: str) -> (list, np.ndarray):
    """Loads raw model output from the on-disk cache if it exists."""
    cache_key_dir = os.path.join(CACHE_DIR, cache_key)
    if not os.path.exists(cache_key_dir):
        return None, None

    try:
        styles = np.load(os.path.join(cache_key_dir, "styles.npy"))

        flows_dir = os.path.join(cache_key_dir, "flows")
        flow_files = sorted(os.listdir(flows_dir))
        flows = [np.load(os.path.join(flows_dir, f)) for f in flow_files]

        logger.debug("Loaded prediction from cache with key %s", cache_key)
        return flows, styles
    except Exception as e:
        logger.warning(
            "Failed to load from cache (key: %s): %s. Will recompute.", cache_key, e
        )
        return None, None
