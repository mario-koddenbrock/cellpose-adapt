import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Cellpose segmentation pipeline."""

    # Model parameters
    model_name: str = "cpsam"
    diameter: float = 30.0
    channel_to_segment: Optional[int] = None  # Channel index to segment, None for all channels

    # Pre-processing parameters
    invert: bool = False
    normalize: bool = True
    norm3D: bool = False
    percentile_min: float = 1.0
    percentile_max: float = 99.0
    sharpen_radius: float = 0.0
    smooth_radius: float = 0.0

    # Evaluation parameters
    do_3D: bool = True
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    max_size_fraction: float = 2.0
    niter: int = 0

    # Tiling and stitching
    tile: bool = False # Whether to use tiling
    tile_overlap: float = 0.1
    stitch_threshold: float = 0.0
    tile_norm_blocksize: int = 0  # Added
    tile_norm_smooth3D: int = 1  # Added

    # Axis configuration
    channel_axis: Optional[int] = None
    z_axis: int = 0

    def to_json(self, file_path: str):
        """Saves the configuration to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, file_path: str):
        """Loads a configuration from a JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self):
        return asdict(self)
