import hashlib
import json
import os
from dataclasses import asdict

import numpy as np


def filter_model_parameter(params):
    model_keys = [
        "channel_axis",
        "channel_nuclei",
        "channel_segment",
        "diameter",
        "do_3D",
        "invert",
        "model_name",
        "norm3D",
        "normalize",
        "percentile_max",
        "percentile_min",
        "sharpen_radius",
        "smooth_radius",
        "stitch_threshold",
        "tile_norm_blocksize",
        "tile_norm_smooth3D",
        "tile_overlap",
    ]
    return {k: params.get(k) for k in model_keys}

def compute_hash(image, parameters, separate_mask_computing:bool = True):
    """
    Compute a unique hash for the image and parameters.
    """
    if separate_mask_computing:
        parameters = filter_model_parameter(parameters)

    image_hash = hashlib.sha256(image.tobytes()).hexdigest()

    if isinstance(parameters, dict):
        param_hash = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    else:
        param_hash = hashlib.sha256(json.dumps(asdict(parameters), sort_keys=True).encode()).hexdigest()

    return f"{image_hash}_{param_hash}"


def save_to_cache(cache_dir, cache_key, masks, flows, styles, diams):
    """
    Save data to the cache directory with separate files for each component.
    """
    cache_key_dir = os.path.join(cache_dir, cache_key)
    if not os.path.exists(cache_key_dir):
        os.makedirs(cache_key_dir)

    # Save each component separately
    np.save(os.path.join(cache_key_dir, "masks.npy"), masks.astype(np.uint16))
    np.save(os.path.join(cache_key_dir, "styles.npy"), styles.astype(np.float32))
    np.save(os.path.join(cache_key_dir, "diams.npy"), np.array([diams], dtype=np.float32))

    # Save flows as separate files
    flows_dir = os.path.join(cache_key_dir, "flows")
    if not os.path.exists(flows_dir):
        os.makedirs(flows_dir)
    for i, flow in enumerate(flows):
        np.save(os.path.join(flows_dir, f"flow_{i}.npy"), flow.astype(np.uint8))


def load_from_cache(cache_dir, cache_key):
    """
    Load data from the cache directory with separate files for each component.
    """
    cache_key_dir = os.path.join(cache_dir, cache_key)
    if not os.path.exists(cache_key_dir):
        return None, None, None, None

    try:
        masks = np.load(os.path.join(cache_key_dir, "masks.npy"))
    except Exception as e:
        print(f"Error: {e}")
        masks = None

    try:
        styles = np.load(os.path.join(cache_key_dir, "styles.npy"))
    except Exception as e:
        print(f"Error: {e}")
        styles = None

    try:
        diams = np.load(os.path.join(cache_key_dir, "diams.npy")).item()
    except Exception as e:
        print(f"Error: {e}")
        diams = None

    try:
        flows_dir = os.path.join(cache_key_dir, "flows")
        flows = [np.load(os.path.join(flows_dir, f)) for f in sorted(os.listdir(flows_dir))]
    except Exception as e:
        print(f"Error: {e}")
        flows = None

    return masks, flows, styles, diams
