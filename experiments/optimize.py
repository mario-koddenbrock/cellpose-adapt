import os
from itertools import product

import wandb

from cellpose_adapt.config import ensure_default_parameter, CellposeConfig
from cellpose_adapt.core import EvaluationError
from cellpose_adapt.file_io import load_image_with_gt
from cellpose_adapt.main import evaluate_model
from cellpose_adapt.viz import show_napari
from experiments.results import ResultHandler


def grid_search(
        params: dict,
        data: list,
        result_file: str = "",
        cache_dir: str = ".cache",
        show_viewer: bool = False,
        log_wandb: bool = False,
        append_result: bool = False,
):

    # Ensure that all images and ground truth files exist
    for image_path, ground_truth_path in data:
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist.")
        if not os.path.exists(ground_truth_path):
            print(f"Labels file {ground_truth_path} does not exist.")

    for image_idx, (image_path, ground_truth_path) in enumerate(data):

        if log_wandb:
            image_name = os.path.basename(image_path).replace(".tif", "")
            result_name = os.path.basename(result_file).replace(".csv", "")
            wandb.init(project="organoid_segmentation", name=f"{image_name}_{result_name}")

        if "nuclei" in ground_truth_path.lower():
            params["type"] = "Nuclei"
        elif "membrane" in ground_truth_path.lower():
            params["type"] = "Membranes"
        else:
            raise ValueError(f"Invalid ground truth: {ground_truth_path}")

        result_handler = ResultHandler(result_file, log_wandb, append_result)
        append_result = True

        params = ensure_default_parameter(params)

        image_name = os.path.basename(image_path).replace(".tif", "")
        image_orig, ground_truth = load_image_with_gt(image_path, ground_truth_path)

        if image_orig is None:
            continue

        if ground_truth is None:
            continue

        print(f"Processing {image_name}")

        if params["type"] == "Nuclei":
            channel_idx = 0
        elif params["type"] == "Membranes":
            channel_idx = 1
        else:
            raise ValueError(f"Invalid type: {params.type}")

        # Get the right channel
        image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig

        # Get all combinations of parameters
        param_combinations = product(
            params["model_name"],
            params["channel_segment"],
            params["channel_nuclei"],
            params["channel_axis"],
            params["invert"],
            params["normalize"],
            params["percentile_min"],
            params["percentile_max"],
            params["diameter"],
            params["do_3D"],
            params["flow_threshold"],
            params["cellprob_threshold"],
            params["interp"],
            params["min_size"],
            params["max_size_fraction"],
            params["niter"],
            params["stitch_threshold"],
            params["tile_overlap"],
            params["norm3D"],
            params["sharpen_radius"],
            params["smooth_radius"],
            params["tile_norm_blocksize"],
            params["tile_norm_smooth3D"],
        )

        for combination in param_combinations:
            (
                model_name, channel_segment, channel_nuclei, channel_axis, invert, normalize,
                percentile_min, percentile_max, diameter, do_3D, flow_threshold, cellprob_threshold,
                interp, min_size, max_size_fraction, niter, stitch_threshold, tile_overlap,
                norm3D, sharpen_radius, smooth_radius, tile_norm_blocksize, tile_norm_smooth3D
            ) = combination

            config = CellposeConfig(
                cellprob_threshold=cellprob_threshold,
                channel_axis=channel_axis,
                channel_nuclei=channel_nuclei,
                channel_segment=channel_segment,
                diameter=diameter,
                do_3D=do_3D,
                flow_threshold=flow_threshold,
                interp=interp,
                invert=invert,
                max_size_fraction=max_size_fraction,
                min_size=min_size,
                model_name=model_name,
                niter=niter,
                norm3D=norm3D,
                normalize=normalize,
                percentile_max=percentile_max,
                percentile_min=percentile_min,
                sharpen_radius=sharpen_radius,
                smooth_radius=smooth_radius,
                stitch_threshold=stitch_threshold,
                tile_norm_blocksize=tile_norm_blocksize,
                tile_norm_smooth3D=tile_norm_smooth3D,
                tile_overlap=tile_overlap,
                type=params["type"],
            )

            results = evaluate_model(image_name, image, ground_truth, config, cache_dir)

            if results == EvaluationError.GROUND_TRUTH_NOT_AVAILABLE:
                break
            elif not isinstance(results, dict):
                continue

            result_handler.log_result(results, config)

            if show_viewer:
                show_napari(results, config)

            # if results["jaccard"] > 0.95:
            #     print(f"Found good parameters for {type} on {image_path}")
            #     return

    if log_wandb:
        wandb.finish()
