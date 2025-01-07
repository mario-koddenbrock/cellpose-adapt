import time
from enum import Enum

import numpy as np
import torch
from cellpose.dynamics import compute_masks
from cellpose.metrics import aggregated_jaccard_index
from cellpose.models import CellposeModel

from .hash import save_to_cache, compute_hash, load_from_cache
from .metrics import jaccard
from .utils import check_set_gpu


def evaluate_model(key, image, ground_truth, params,
                   cache_dir=".cache", separate_mask_computing=True, only_cached_results=False):
    t0 = time.time()


    cache_key = compute_hash(image, params, separate_mask_computing)

    masks, flows, styles, diams = load_from_cache(cache_dir, cache_key)
    model_name = params.model_name

    if masks is not None:
        print(f"\tLOADING FROM CACHE: {model_name}")
    else:

        if only_cached_results:
            return EvaluationError.CACHE_NOT_AVAILABLE

        print(f"\tEVALUATING: {model_name}")
        try:

            device = check_set_gpu()
            model = CellposeModel(device=device, gpu=False, model_type=params.model_name,
                                    diam_mean=params.diameter, nchan=2,
                                    backbone="default")

            #     Args:
            #         normalize (bool, optional): Whether to perform normalization. Defaults to True.
            #         norm3D (bool, optional): Whether to normalize in 3D. If True, the entire 3D stack will
            #             be normalized per channel. If False, normalization is applied per Z-slice. Defaults to False.
            #         invert (bool, optional): Whether to invert the image. Useful if cells are dark instead of bright.
            #             Defaults to False.
            #         lowhigh (tuple or ndarray, optional): The lower and upper bounds for normalization.
            #             Can be a tuple of two values (applied to all channels) or an array of shape (nchan, 2)
            #             for per-channel normalization. Incompatible with smoothing and sharpening.
            #             Defaults to None.
            #         percentile (tuple, optional): The lower and upper percentiles for normalization. If provided, it should be
            #             a tuple of two values. Each value should be between 0 and 100. Defaults to (1.0, 99.0).
            #         sharpen_radius (int, optional): The radius for sharpening the image. Defaults to 0.
            #         smooth_radius (int, optional): The radius for smoothing the image. Defaults to 0.
            #         tile_norm_blocksize (int, optional): The block size for tile-based normalization. Defaults to 0.
            #         tile_norm_smooth3D (int, optional): The smoothness factor for tile-based normalization in 3D. Defaults to 1.
            #         axis (int, optional): The channel axis to loop over for normalization. Defaults to -1.

            normalzation_params = {
                "invert": params.invert,
                "lowhigh": None,
                "norm3D": params.norm3D,
                "normalize": params.normalize,
                "percentile": (float(params.percentile_min), float(params.percentile_max)),
                "sharpen_radius": params.sharpen_radius,
                "smooth_radius": params.smooth_radius,
                "tile_norm_blocksize": params.tile_norm_blocksize,
                "tile_norm_smooth3D": params.tile_norm_smooth3D,
            }

            masks, flows, styles = model.eval(
                image,
                cellprob_threshold=params.cellprob_threshold,
                channel_axis=params.channel_axis,
                channels=[params.channel_segment, params.channel_nuclei],
                compute_masks=(not separate_mask_computing),
                diameter=params.diameter,
                do_3D=params.do_3D,
                flow_threshold=params.flow_threshold,
                invert=params.invert,
                interp=params.interp,
                max_size_fraction=params.max_size_fraction,
                min_size=params.min_size,
                niter=params.niter,
                normalize=normalzation_params,
                tile_overlap=params.tile_overlap,
                stitch_threshold=params.stitch_threshold,
                z_axis=0,
            )

            print(f"\tMask: {np.any(masks)}")
            save_to_cache(cache_dir, cache_key, masks, flows, styles, params.diameter)

        except Exception as e:
            print(f"Error: {e}")
            return EvaluationError.EVALUATION_ERROR

    if separate_mask_computing:
        # dP_colors = flows[0]
        dP = flows[1]
        cellprob = flows[2]

        # params.min_size = 60000

        masks = compute_masks(
            dP,
            cellprob,
            niter=params.niter,
            cellprob_threshold=params.cellprob_threshold,
            flow_threshold=params.flow_threshold,
            interp=params.interp,
            do_3D=params.do_3D,
            min_size=params.min_size,
            max_size_fraction=params.max_size_fraction,
            device=torch.device('cpu'), # TODO: MPS not available
        )

    if masks is None:
        print(f"Error: No masks found with parameters")
        return EvaluationError.EMPTY_MASKS

    else:
        jaccard_score = jaccard(ground_truth, masks)
        
        aji_scores = aggregated_jaccard_index([ground_truth], [masks])
        jaccard_cellpose = np.mean(aji_scores[~np.isnan(aji_scores)])
        # jaccard_score = jaccard_cellpose

        print(f"\tJaccard (own): {jaccard_score:.2f}")
        print(f"\tJaccard (cellpose): {jaccard_cellpose:.2f}")


    results = {
        "image": image,
        "image_name": key,
        "ground_truth": ground_truth,
        "masks": masks,
        "jaccard": jaccard_score,
        "jaccard_cellpose": jaccard_cellpose,
        "duration": time.time() - t0,
    }

    return results


class EvaluationError(Enum):
    CACHE_NOT_AVAILABLE = "Cache not available"
    EMPTY_MASKS = "Empty masks"
    EVALUATION_ERROR = "Evaluation error"
    GROUND_TRUTH_NOT_AVAILABLE = "Ground truth not available"
    IMAGE_NOT_AVAILABLE = "Image not available"
