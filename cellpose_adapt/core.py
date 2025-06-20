import logging
import time

import numpy as np
import torch
from cellpose.dynamics import compute_masks
from cellpose.models import CellposeModel

from . import caching
from .config.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)
logger.debug("Cellpose core module loaded.")

def initialize_model(model_name: str, device: torch.device) -> CellposeModel:
    """Initializes and returns a CellposeModel instance on a specific device."""
    logger.info("Initializing Cellpose model '%s' on device '%s'", model_name, device)
    try:
        model = CellposeModel(
            gpu=device.type != 'cpu',
            pretrained_model=model_name,
            device=device
        )
        return model
    except Exception as e:
        logger.error("Failed to initialize Cellpose model: %s", e)
        raise

class CellposeRunner:
    """Encapsulates the Cellpose evaluation logic for a given config and preloaded model."""
    def __init__(self, model: CellposeModel, config: PipelineConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device

    def _get_raw_output(self, image: np.ndarray) -> (list, np.ndarray):
        """
        Gets the raw model output (flows, styles), using a cache to avoid re-computation.
        This is the slow part of the pipeline.
        """
        model_params = caching.get_model_eval_params(self.config)
        cache_key = caching.compute_hash(image, model_params)

        # Try to load from cache first
        cached_flows, cached_styles = caching.load_from_cache(cache_key)
        if cached_flows is not None:
            logger.info("CACHE HIT for model prediction.")
            return cached_flows, cached_styles

        # If not in cache, run the model
        logger.info("CACHE MISS. Running model.eval() to get raw flows.")

        # Build the full normalization dictionary
        normalization_params = {
            "normalize": self.config.normalize,
            "norm3D": self.config.norm3D,
            "invert": self.config.invert,
            "percentile": (self.config.percentile_min, self.config.percentile_max),
            "sharpen_radius": self.config.sharpen_radius,
            "smooth_radius": self.config.smooth_radius,
            "tile_norm_blocksize": self.config.tile_norm_blocksize,  # Added
            "tile_norm_smooth3D": self.config.tile_norm_smooth3D,  # Added
        }

        masks, flows, styles = self.model.eval(
            x=image,
            diameter=self.config.diameter if self.config.diameter > 0 else None,
            do_3D=self.config.do_3D,
            normalize=normalization_params,  # Pass the full dictionary
            compute_masks=False,
            tile_overlap=self.config.tile_overlap,
            stitch_threshold=self.config.stitch_threshold,
            z_axis=self.config.z_axis,
            resample=True,
            channel_axis=self.config.channel_axis,
            invert=self.config.invert,
            rescale=None,
            flow_threshold=self.config.flow_threshold,
            cellprob_threshold=self.config.cellprob_threshold,
            anisotropy=None,
            flow3D_smooth=0,
            min_size=self.config.min_size,
            max_size_fraction=self.config.max_size_fraction,
            niter=self.config.niter,
            augment=False,
            bsize=256,
        )

        # Save the new result to the cache
        caching.save_to_cache(cache_key, flows, styles)

        return flows, styles

    def run(self, image: np.ndarray) -> (np.ndarray, float):
        """
        Runs the full segmentation pipeline on a single image.

        1. Get raw model output (from cache or by running the model).
        2. Computes final masks using post-processing parameters.
        """
        t0 = time.time()

        try:
            # Step 1: Get raw output (cached)
            flows, _ = self._get_raw_output(image)

            # Step 2: Compute masks from raw output (fast)
            # This is where post-processing hyperparameters are used.
            dP, cellprob = flows[1], flows[2]

            masks = compute_masks(
                dP,
                cellprob,
                niter=(
                    self.config.niter
                    if self.config.niter > 0
                    else (200 if image.ndim > 2 else 0)
                ),
                flow_threshold=self.config.flow_threshold,
                cellprob_threshold=self.config.cellprob_threshold,
                min_size=self.config.min_size,
                do_3D=self.config.do_3D,
                max_size_fraction=self.config.max_size_fraction,
                device=self.device,
            )

            duration = time.time() - t0
            logger.info(
                "Segmentation finished in %.2f seconds. Found %d masks.",
                duration,
                len(np.unique(masks)) - 1,
            )
            return masks, duration

        except Exception as e:
            logger.error("Error during Cellpose evaluation: %s", e, exc_info=True)
            return None, time.time() - t0


