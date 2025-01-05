from dataclasses import dataclass

import yaml


# The dataset-specific models were trained on the training images from the following datasets:
# tissuenet_cp3: tissuenet dataset.
# livecell_cp3: livecell dataset
# yeast_PhC_cp3: YEAZ dataset
# yeast_BF_cp3: YEAZ dataset
# bact_phase_cp3: omnipose dataset
# bact_fluor_cp3: omnipose dataset
# deepbacs_cp3: deepbacs dataset
# cyto2_cp3: cellpose dataset

# We have a nuclei model and a super-generalist cyto3 model.
# There are also two older models, cyto, which is trained on only the Cellpose training set, and cyto2, which is also trained on user-submitted images.

# nchan (int, optional): Number of channels to use as input to network, default is 2 (cyto + nuclei) or (nuclei + zeros). TODO
# backbone_list # TODO
# anisotropic_list # TODO

available_model_list = [
    "cyto3",
    "cyto2_cp3",
    "cyto",
    "cyto2",
    "nuclei",
    "tissuenet_cp3",
    "livecell_cp3",
    "yeast_PhC_cp3",
    "yeast_BF_cp3",
    "bact_phase_cp3",
    "bact_fluor_cp3",
    "deepbacs_cp3",
]

def ensure_default_parameter(params):
    default_params = {
        "cellprob_threshold": [5],
        "channel_axis": [None],
        "channel_nuclei": [0],  # apparently, this is not used in grayscale
        "channel_segment": [0],  # apparently, this is not used in grayscale
        "diameter": [1000],
        "do_3D": [True],
        "flow_threshold": [0.5],  # apparently, this is not used in 3D
        "interp": [False],
        "invert": [False],
        "max_size_fraction": [0.5],
        "min_size": [20000],
        "model_name": ["cyto3"],
        "niter": [100],
        "norm3D": [False],
        "normalize": [True],
        "percentile_max": [99.0],
        "percentile_min": [0.0],
        "sharpen_radius": [0.0],
        "smooth_radius": [1.0],
        "stitch_threshold": [0.75],
        "tile_norm_blocksize": [0],
        "tile_norm_smooth3D": [1],
        "tile_overlap": [0.0],
    }

    diameter_not_set = "diameter" not in params
    for key in default_params.keys():
        if key not in params:
            params[key] = default_params[key]

    if diameter_not_set:
        diam_mean = 17. if "nuclei" in params["type"][0].lower() else 30.
        params["diameter"] = [diam_mean]

    return params

# Define a dataclass to store the evaluation parameters
@dataclass
class CellposeConfig:
    cellprob_threshold: float
    channel_axis: any
    channel_nuclei: int
    channel_segment: int
    diameter: int
    do_3D: bool
    flow_threshold: float
    interp: bool
    invert: bool
    max_size_fraction: float
    min_size: int
    model_name: str
    niter: int
    norm3D: bool
    normalize: bool
    percentile_max: int
    percentile_min: int
    sharpen_radius: int
    smooth_radius: int
    stitch_threshold: float
    tile_norm_blocksize: int
    tile_norm_smooth3D: int
    tile_overlap: float
    type: str

    def to_yaml(self, yaml_file: str):
        """
        Save an CellposeConfig object to a YAML file.

        Parameters:
            self (CellposeConfig): The object containing evaluation parameters.
            yaml_file (str): The path to save the YAML file.
        """
        config = {
            "cellprob_threshold": self.cellprob_threshold,
            "channel_axis": self.channel_axis,
            "channel_nuclei": self.channel_nuclei,
            "channel_segment": self.channel_segment,
            "diameter": self.diameter,
            "do_3D": self.do_3D,
            "flow_threshold": self.flow_threshold,
            "interp": self.interp,
            "invert": self.invert,
            "max_size_fraction": self.max_size_fraction,
            "min_size": self.min_size,
            "model_name": self.model_name,
            "niter": self.niter,
            "norm3D": self.norm3D,
            "normalize": self.normalize,
            "percentile_max": self.percentile_max,
            "percentile_min": self.percentile_min,
            "sharpen_radius": self.sharpen_radius,
            "smooth_radius": self.smooth_radius,
            "stitch_threshold": self.stitch_threshold,
            "tile_norm_blocksize": self.tile_norm_blocksize,
            "tile_norm_smooth3D": self.tile_norm_smooth3D,
            "tile_overlap": self.tile_overlap,
            "type": self.type,
        }
        try:
            with open(yaml_file, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            print(f"Saved CellposeConfig to {yaml_file}")
        except Exception as e:
            print(f"Error saving CellposeConfig to YAML: {e}")


    def get(self, key, default=None):
        return getattr(self, key, default)

    def asdict(self):
        return self.__dict__
