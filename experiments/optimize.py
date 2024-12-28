import os

import wandb
from sympy.physics.vector.printing import params

from cellpose_adapt.file_io import load_image_with_gt
from cellpose_adapt.utils import check_set_gpu
from experiments.results import ResultHandler
from cellpose_adapt.config import ensure_default_parameter, CellposeConfig
from cellpose_adapt.core import EvaluationError
from cellpose_adapt.main import evaluate_model
from cellpose_adapt.viz import show_napari


def optimize_parameters(
        params: dict,
        data: list,
        result_file: str = "",
        cache_dir: str = ".cache",
        show_viewer: bool = False,
        log_wandb: bool = False,
        append_result: bool = False,
):


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

        for model_name in params["model_name"]:
            for channel_segment in params["channel_segment"]:
                for channel_nuclei in params["channel_nuclei"]:
                    for channel_axis in params["channel_axis"]:
                        for invert in params["invert"]:
                            for normalize in params["normalize"]:
                                for percentile_min in params["percentile_min"]:
                                    for percentile_max in params["percentile_max"]:
                                        for diameter in params["diameter"]:
                                            for do_3D in params["do_3D"]:
                                                for flow_threshold in params["flow_threshold"]:
                                                    for cellprob_threshold in params["cellprob_threshold"]:
                                                        for interp in params["interp"]:
                                                            for min_size in params["min_size"]:
                                                                for max_size_fraction in params["max_size_fraction"]:
                                                                    for niter in params["niter"]:
                                                                        for stitch_threshold in params["stitch_threshold"]:
                                                                            for tile_overlap in params["tile_overlap"]:
                                                                                for norm3D in params["norm3D"]:
                                                                                    for sharpen in params["sharpen"]:
                                                                                        for tile_norm in params["tile_norm"]:

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
                                                                                                sharpen=sharpen,
                                                                                                stitch_threshold=stitch_threshold,
                                                                                                tile_norm=tile_norm,
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
