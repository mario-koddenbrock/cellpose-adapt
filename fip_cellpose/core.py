import os
import time
from enum import Enum

import numpy as np
from cellpose import transforms
from cellpose.metrics import aggregated_jaccard_index
from cellpose.models import CellposeModel
from skimage.metrics import adapted_rand_error

from .file_io import load_image_with_gt
from .hash import save_to_cache, compute_hash, load_from_cache
from .metrics import jaccard
from .utils import check_set_gpu


def evaluate_model(image_path, params, cache_dir=".cache", compute_masks=True):
    t0 = time.time()

    image_name = os.path.basename(image_path).replace(".tif", "")
    device = check_set_gpu()  # get available torch device (CPU, GPU or MPS)
    ground_truth, image_orig = load_image_with_gt(image_path, params.type)

    if ground_truth is None:
        return EvaluationError.GROUND_TRUTH_NOT_AVAILABLE

    print(f"Processing {image_name} ({device})")

    if params.type == "Nuclei":
        channel_idx = 0
    elif params.type == "Membranes":
        channel_idx = 1
    else:
        raise ValueError(f"Invalid type: {params.type}")

    # Get the right channel
    image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig

    # get intensity percentile for normalization
    q1, q3 = np.percentile(image, [params.normalization_min, params.normalization_max])
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

            # return EvaluationError.EVALUATION_ERROR

            model = CellposeModel(device=device, gpu=False, model_type=params.model_name,
                                    diam_mean=params.diameter, nchan=2,
                                    backbone="default")

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
                max_size_fraction=0.5, # default
                min_size=15, # default
                normalize=params.normalize,
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

        masks = model.cp._compute_masks(
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
        "image_name": image_name,
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
    EVALUATION_ERROR = "Evaluation error"
    EMPTY_MASKS = "Empty masks"
