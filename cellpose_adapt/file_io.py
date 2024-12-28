import os.path

import yaml
from cachetools import cached, TTLCache
from cellpose import io

from .config import CellposeConfig

cache = TTLCache(maxsize=100, ttl=300)


@cached(cache)
def load_image_with_gt(image_path, ground_truth_path):

    if os.path.exists(image_path):
        image = io.imread(image_path)
    else:
        print(f"Image not found: {image_path}. Skipping.")
        image = None

    if os.path.exists(ground_truth_path):
        ground_truth = io.imread(ground_truth_path)
    else:
        print(f"Ground truth not found: {ground_truth_path}. Skipping.")
        ground_truth = None

    return image, ground_truth


# def get_cellpose_ground_truth(image_path, image_name, type="Nuclei"):
#
#     ground_truth_path = image_path.replace("images_cropped_isotropic", f"labelmaps/{type}")
#     ground_truth_path = ground_truth_path.replace(".tif", f"_{type.lower()}-labels.tif")
#
#     if os.path.exists(ground_truth_path):
#         ground_truth = io.imread(ground_truth_path)
#     else:
#         print(f"Ground truth not found for {image_name}. Skipping.")
#         ground_truth = None
#     return ground_truth


def read_yaml(yaml_file: str = "") -> CellposeConfig:
    """
    Read a YAML file containing model configuration and return an CellposeConfig object.

    Parameters:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        CellposeConfig: An object populated with the parameters from the YAML file.
    """
    # Load configuration from YAML
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{yaml_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None

    print(f"Loaded configuration from {yaml_file}:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Map configuration to CellposeConfig
    params = CellposeConfig(
        model_name=config.get('model_name'),
        channel_segment=config.get('channel_segment'),
        channel_nuclei=config.get('channel_nuclei'),
        channel_axis=config.get('channel_axis'),
        invert=config.get('invert'),
        normalize=config.get('normalize'),
        percentile_min=config.get('percentile_min'),
        percentile_max=config.get('percentile_max'),
        diameter=config.get('diameter'),
        do_3D=config.get('do_3D'),
        flow_threshold=config.get('flow_threshold'),
        cellprob_threshold=config.get('cellprob_threshold'),
        interp=config.get('interp'),
        min_size=config.get('min_size'),
        max_size_fraction=config.get('max_size_fraction'),
        niter=config.get('niter'),
        stitch_threshold=config.get('stitch_threshold'),
        tile_overlap=config.get('tile_overlap'),
        type=config.get('type')
    )

    return params
