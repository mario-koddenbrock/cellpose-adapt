import os

import wandb

from experiments.results import ResultHandler
from cellpose_adapt.config import ensure_default_parameter, CellposeConfig
from cellpose_adapt.core import EvaluationError
from cellpose_adapt.main import evaluate_model
from cellpose_adapt.viz import show_napari


def optimize_parameters(
        param_options: dict,
        image_path: str = "",
        result_file: str = "",
        cache_dir: str = ".cache",
        show_viewer: bool = False,
        log_wandb: bool = True,
        append_result: bool = False,
):

    if log_wandb:
        image_name = os.path.basename(image_path).replace(".tif", "")
        result_name = os.path.basename(result_file).replace(".csv", "")
        wandb.init(project="organoid_segmentation", name=f"{image_name}_{result_name}")

    result_handler = ResultHandler(result_file, log_wandb, append_result)

    param_options = ensure_default_parameter(param_options)

    for model_name in param_options["model_name"]:
        for channel_segment in param_options["channel_segment"]:
            for channel_nuclei in param_options["channel_nuclei"]:
                for channel_axis in param_options["channel_axis"]:
                    for invert in param_options["invert"]:
                        for normalize in param_options["normalize"]:
                            for normalization_min in param_options["normalization_min"]:
                                for normalization_max in param_options["normalization_max"]:
                                    for diameter in param_options["diameter"]:
                                        for do_3D in param_options["do_3D"]:
                                            for flow_threshold in param_options["flow_threshold"]:
                                                for cellprob_threshold in param_options["cellprob_threshold"]:
                                                    for interp in param_options["interp"]:
                                                        for min_size in param_options["min_size"]:
                                                            for max_size_fraction in param_options["max_size_fraction"]:
                                                                for niter in param_options["niter"]:
                                                                    for stitch_threshold in param_options["stitch_threshold"]:
                                                                        for tile_overlap in param_options["tile_overlap"]:
                                                                            for type in param_options["type"]:

                                                                                params = CellposeConfig(
                                                                                    model_name=model_name,
                                                                                    channel_segment=channel_segment,
                                                                                    channel_nuclei=channel_nuclei,
                                                                                    channel_axis=channel_axis,
                                                                                    invert=invert,
                                                                                    normalize=normalize,
                                                                                    normalization_min=normalization_min,
                                                                                    normalization_max=normalization_max,
                                                                                    diameter=diameter,
                                                                                    do_3D=do_3D,
                                                                                    flow_threshold=flow_threshold,
                                                                                    cellprob_threshold=cellprob_threshold,
                                                                                    interp=interp,
                                                                                    min_size=min_size,
                                                                                    max_size_fraction=max_size_fraction,
                                                                                    niter=niter,
                                                                                    stitch_threshold=stitch_threshold,
                                                                                    tile_overlap=tile_overlap,
                                                                                    type=type,
                                                                                )

                                                                                results = evaluate_model(image_path, params, cache_dir)

                                                                                if results == EvaluationError.GROUND_TRUTH_NOT_AVAILABLE:
                                                                                    break
                                                                                elif not isinstance(results, dict):
                                                                                    continue

                                                                                result_handler.log_result(results, params)

                                                                                if show_viewer:
                                                                                    show_napari(results, params)

                                                                                if results["jaccard"] > 0.95:
                                                                                    print(f"Found good parameters for {type} on {image_path}")
                                                                                    return

    if log_wandb:
        wandb.finish()
