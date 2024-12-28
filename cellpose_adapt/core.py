import time
from enum import Enum

import numpy as np
from cellpose import transforms
from cellpose.metrics import aggregated_jaccard_index
from cellpose.models import CellposeModel
from skimage.metrics import adapted_rand_error

from .hash import save_to_cache, compute_hash, load_from_cache
from .metrics import jaccard
from .utils import check_set_gpu


def evaluate_model(key, image, ground_truth, params, cache_dir=".cache", compute_masks=True):
    t0 = time.time()

    # get intensity percentile for normalization
    q1, q3 = np.percentile(image, [params.percentile_min, params.percentile_max])
    image = np.clip(image, q1, q3)

    # plot the intensity distribution of the image
    # plot_intensity(image)

    cache_key = compute_hash(image, params, compute_masks)

    cached_result = load_from_cache(cache_dir, cache_key)
    model_name = params.model_name

    if cached_result:
        print(f"\tLOADING FROM CACHE: {model_name}")
        masks, flows, styles, diams = cached_result
    else:
        print(f"\tEVALUATING: {model_name}")
        try:

            # return {
            #     "image": image,
            #     "image_name": key,
            #     "ground_truth": ground_truth,
            #     "masks": None,
            #     "are": 42,
            #     "precision": 43,
            #     "recall": 44,
            #     "f1": 45,
            #     "jaccard": 46,
            #     "jaccard_cellpose": 47,
            #     "duration": 100,
            # }

            device = check_set_gpu()
            model = CellposeModel(device=device, gpu=False, model_type=params.model_name,
                                    diam_mean=params.diameter, nchan=2,
                                    backbone="default")

            #     Args:
            #         img (ndarray): The input image. It should have at least 3 dimensions.
            #             If it is 4-dimensional, it assumes the first non-channel axis is the Z dimension.
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
                "lowhigh": None,
                "norm3D": params.norm3D,
                "normalize": params.normalize,
                "invert": params.invert,
                "percentile": (params.percentile_min, params.percentile_max),
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
                compute_masks=compute_masks,
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
                z_axis=0,  # TODO: z-axis parameter always 0?
            )
            save_to_cache(cache_dir, cache_key, masks, flows, styles, params.diameter)

        except Exception as e:
            print(f"Error: {e}")
            return EvaluationError.EVALUATION_ERROR

    if not compute_masks:
        # dP_colors = flows[0]
        dP = flows[1]
        cellprob = flows[2]
        nchan = 2  # TODO
        x = transforms.convert_image(
            image,
            [params.channel_segment, params.channel_nuclei],
            channel_axis=params.channel_axis,
            z_axis=0,
            do_3D=(params.do_3D or params.stitch_threshold > 0),
            nchan=nchan)

        # TODO model might be not initialized

        masks = model._compute_masks(
            x.shape, dP, cellprob,
            flow_threshold=params.flow_threshold,
            cellprob_threshold=params.cellprob_threshold,
            interp=params.interp,
            min_size=params.min_size,
            max_size_fraction=params.max_size_fraction,
            niter=params.niter,
            stitch_threshold=params.stitch_threshold,
            do_3D=params.do_3D,
        )

        masks = masks.squeeze()

    if masks is None:
        print(f"Error: No masks found with parameters")
        return EvaluationError.EMPTY_MASKS

    else:
        jaccard_score = jaccard(ground_truth, masks)
        
        aji_scores = aggregated_jaccard_index([ground_truth], [masks])
        jaccard_cellpose = np.mean(aji_scores[~np.isnan(aji_scores)])

        are, precision, recall = adapted_rand_error(ground_truth, masks)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        print(f"\tAdapted Rand Error: {are:.2f}")
        print(f"\tPrecision: {precision:.2f}")
        print(f"\tRecall: {recall:.2f}")
        print(f"\tF1: {f1:.2f}")
        print(f"\tJaccard (own): {jaccard_score:.2f}")
        print(f"\tJaccard (cellpose): {jaccard_cellpose:.2f}")


    duration = time.time() - t0

    results = {
        "image": image,
        "image_name": key,
        "ground_truth": ground_truth,
        "masks": masks,
        "are": are,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard_score,
        "jaccard_cellpose": jaccard_cellpose,
        "duration": duration,
    }

    return results





class EvaluationError(Enum):
    GROUND_TRUTH_NOT_AVAILABLE = "Ground truth not available"
    IMAGE_NOT_AVAILABLE = "Image not available"
    EVALUATION_ERROR = "Evaluation error"
    EMPTY_MASKS = "Empty masks"
