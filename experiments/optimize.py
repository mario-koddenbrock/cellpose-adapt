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
        log_wandb: bool = False,
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

                                                                                # if results["jaccard"] > 0.95:
                                                                                #     print(f"Found good parameters for {type} on {image_path}")
                                                                                #     return

    if log_wandb:
        wandb.finish()









#
# import os
# import itertools
# import wandb
# from concurrent.futures import ProcessPoolExecutor
# from experiments.results import ResultHandler
# from cellpose_adapt.config import ensure_default_parameter, CellposeConfig
# from cellpose_adapt.core import EvaluationError
# from cellpose_adapt.main import evaluate_model
# from cellpose_adapt.viz import show_napari
#
#
# def evaluate_params(image_path, result_file, cache_dir, show_viewer, log_wandb, append_result, param_combination):
#     """Evaluates a single combination of parameters."""
#     params = CellposeConfig(**param_combination)
#     result_handler = ResultHandler(result_file, log_wandb, append_result)
#
#     try:
#         results = evaluate_model(image_path, params, cache_dir)
#         if results == EvaluationError.GROUND_TRUTH_NOT_AVAILABLE:
#             return None
#         elif not isinstance(results, dict):
#             return None
#
#         result_handler.log_result(results, params)
#
#         if show_viewer:
#             show_napari(results, params)
#
#         # if results.get("jaccard", 0) > 0.95:
#         #     print(f"Found good parameters: {param_combination}")
#         #     return param_combination
#
#     except Exception as e:
#         print(f"Error with parameters {param_combination}: {e}")
#
#     return None
#
#
# def generate_param_combinations(param_options):
#     """Generates all combinations of parameters."""
#     keys = param_options.keys()
#     combinations = list(itertools.product(*param_options.values()))
#     return [dict(zip(keys, combo)) for combo in combinations]
#
#
# def optimize_parameters(
#         param_options: dict,
#         image_path: str = "",
#         result_file: str = "",
#         cache_dir: str = ".cache",
#         show_viewer: bool = False,
#         log_wandb: bool = False,
#         append_result: bool = False,
#         max_workers: int = 0,  # Number of parallel processes
# ):
#
#     if max_workers == 0:
#         max_workers = os.cpu_count()
#
#     if log_wandb:
#         image_name = os.path.basename(image_path).replace(".tif", "")
#         result_name = os.path.basename(result_file).replace(".csv", "")
#         wandb.init(project="organoid_segmentation", name=f"{image_name}_{result_name}")
#
#     param_options = ensure_default_parameter(param_options)
#     param_combinations = generate_param_combinations(param_options)
#
#     # Parallel execution
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(
#                 evaluate_params,
#                 image_path,
#                 result_file,
#                 cache_dir,
#                 show_viewer,
#                 log_wandb,
#                 append_result,
#                 combination,
#             )
#             for combination in param_combinations
#         ]
#
#         # for future in futures:
#         #     result = future.result()
#         #     if result is not None:
#         #         print(f"Optimal parameters found: {result}")
#         #         break  # Stop further evaluations if optimal parameters are found
#
#     if log_wandb:
#         wandb.finish()
