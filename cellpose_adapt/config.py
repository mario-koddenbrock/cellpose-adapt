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

available_model_list = ["cyto2_cp3", "cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3",
                           "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3"]

def ensure_default_parameter(params):
    default_params = {
        "model_name": ["cyto3"],
        "channel_segment": [0],
        "channel_nuclei": [0],
        "channel_axis": [None],
        "invert": [False],
        "normalize": [True],
        "normalization_min": [0],
        "normalization_max": [1],
        "diameter": [30],
        "do_3D": [True],
        "flow_threshold": [0.1],  # apparently, this is not used in 3D
        "cellprob_threshold": [0.0],
        "interp": [False],
        "min_size": [15],
        "max_size_fraction": [0.5],
        "niter": [100],
        "stitch_threshold": [0.0],
        "tile_overlap": [0.1],
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
    model_name: str
    channel_segment: int
    channel_nuclei: int
    channel_axis: any
    invert: bool
    normalize: bool
    normalization_min: int
    normalization_max: int
    diameter: int
    do_3D: bool
    flow_threshold: float
    cellprob_threshold: float
    interp: bool
    min_size: int
    max_size_fraction: float
    niter: int
    stitch_threshold: float
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
            "model_name": self.model_name,
            "channel_segment": self.channel_segment,
            "channel_nuclei": self.channel_nuclei,
            "channel_axis": self.channel_axis,
            "invert": self.invert,
            "normalize": self.normalize,
            "normalization_min": self.normalization_min,
            "normalization_max": self.normalization_max,
            "diameter": self.diameter,
            "do_3D": self.do_3D,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "interp": self.interp,
            "min_size": self.min_size,
            "max_size_fraction": self.max_size_fraction,
            "niter": self.niter,
            "stitch_threshold": self.stitch_threshold,
            "tile_overlap": self.tile_overlap,
            "type": self.type
        }
        try:
            with open(yaml_file, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            print(f"Saved CellposeConfig to {yaml_file}")
        except Exception as e:
            print(f"Error saving CellposeConfig to YAML: {e}")
