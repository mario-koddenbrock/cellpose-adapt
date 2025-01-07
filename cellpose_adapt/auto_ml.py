import os

import ray
import wandb
from ray import tune
from ray.air import session

from cellpose_adapt.config import CellposeConfig
from cellpose_adapt.core import evaluate_model
from cellpose_adapt.file_io import load_image_with_gt
from cellpose_adapt.utils import set_all_seeds
from experiments.data import data



set_all_seeds(42)


# Wrapping your evaluate_model to integrate with Ray Tune
def objective(params, image_path, ground_truth_path, log_wandb=False):
    key = os.path.basename(image_path).replace(".tif", "")

    image_name = os.path.basename(image_path).replace(".tif", "")
    image_orig, ground_truth = load_image_with_gt(image_path, ground_truth_path)

    if image_orig is None:
        return
    if ground_truth is None:
        return

    # Adjust parameters for type based on ground truth
    if "nuclei" in ground_truth_path.lower():
        params["type"] = "Nuclei"
        channel_idx = 0
    elif "membrane" in ground_truth_path.lower():
        params["type"] = "Membranes"
        channel_idx = 1
    else:
        raise ValueError(f"Invalid ground truth: {ground_truth_path}")

    config = CellposeConfig(
        cellprob_threshold=params["cellprob_threshold"],
        channel_axis=params["channel_axis"],
        channel_nuclei=params["channel_nuclei"],
        channel_segment=params["channel_segment"],
        diameter=params["diameter"],
        do_3D=params["do_3D"],
        flow_threshold=params["flow_threshold"],
        interp=params["interp"],
        invert=params["invert"],
        max_size_fraction=params["max_size_fraction"],
        min_size=params["min_size"],
        model_name=params["model_name"],
        niter=params["niter"],
        norm3D=params["norm3D"],
        normalize=params["normalize"],
        percentile_max=params["percentile_max"],
        percentile_min=params["percentile_min"],
        sharpen_radius=params["sharpen_radius"],
        smooth_radius=params["smooth_radius"],
        stitch_threshold=params["stitch_threshold"],
        tile_norm_blocksize=params["tile_norm_blocksize"],
        tile_norm_smooth3D=params["tile_norm_smooth3D"],
        tile_overlap=params["tile_overlap"],
        type=params["type"],
    )

    # Get the right channel
    image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig

    try:
        if log_wandb:
            # Log each experiment in Weights & Biases
            result_name = f"{key}_optimization"
            wandb.init(project="organoid_segmentation", name=f"{key}_{result_name}")

        # Evaluate the model
        results = evaluate_model(
            key,
            image,
            ground_truth,
            config,
            cache_dir=".cache",
            separate_mask_computing=True,
            only_cached_results=False,
        )

        if not isinstance(results, dict):
            session.report({"jaccard_cellpose": -1.0})

        # Log results to Weights & Biases
        if log_wandb:
            wandb.log({"jaccard_cellpose": results["jaccard_cellpose"]})

        # Report the metric (new session.report API)
        session.report({"jaccard_cellpose": results["jaccard_cellpose"]})

    except Exception as e:
        # Handle errors by reporting a very low score
        session.report({"jaccard_cellpose": -1.0})
        print(f"Error during evaluation: {e}")


def main(data, log_wandb=False, root="../"):
    ray.init(ignore_reinit_error=True, local_mode=False)

    # Ensure all images and ground truth files exist
    for image_path, ground_truth_path in data:
        if not os.path.exists(os.path.join(root, image_path)):
            raise FileNotFoundError(
                f"Image file {os.path.join(root, image_path)} does not exist."
            )
        if not os.path.exists(os.path.join(root, ground_truth_path)):
            raise FileNotFoundError(
                f"Labels file {os.path.join(root, ground_truth_path)} does not exist."
            )

    available_model_list = [
        "cyto3",
        "cyto2_cp3",
        "cyto",
        "cyto2",
        "nuclei",
        "tissuenet_cp3",
        "livecell_cp3",
        "yeast_PhC_cp3",
        "yeast_BF_cp3",
        "bact_phase_cp3",
        "bact_fluor_cp3",
        "deepbacs_cp3",
    ]

    # Define the search space based on experiments
    search_space = {
        "model_name": tune.choice(available_model_list),
        "diameter": tune.uniform(1, 100000),
        "invert": tune.choice([True, False]),
        "norm3D": tune.choice([True, False]),
        "normalize": tune.choice([True, False]),
        "percentile_min": tune.uniform(0.0, 10.0),
        "percentile_max": tune.uniform(90.0, 100.0),
        "sharpen_radius": tune.uniform(0, 500),
        "smooth_radius": tune.uniform(0, 500),
        "tile_norm_blocksize": tune.uniform(0, 500),
        "tile_norm_smooth3D": tune.uniform(0, 500),
        "cellprob_threshold": tune.uniform(-30, 30),
        "channel_axis": tune.choice([None]),
        "channel_segment": tune.choice([0]),
        "channel_nuclei": tune.choice([0]),
        "do_3D": tune.choice([True, False]),
        "flow_threshold": tune.uniform(0, 1),
        "interp": tune.choice([True, False]),
        "max_size_fraction": tune.uniform(0.1, 1.0),
        "min_size": tune.uniform(0, 100000),
        "niter": tune.randint(50, 500),
        "tile_overlap": tune.uniform(0.0, 0.5),
        "stitch_threshold": tune.uniform(0.0, 1.0),
    }

    # Iterate over all image-ground_truth pairs
    for image_idx, (image_path, ground_truth_path) in enumerate(data):
        print(f"Optimizing parameters for image {image_idx + 1}/{len(data)}: {image_path}")

        root = os.path.abspath(root)
        full_image_path = os.path.join(root, image_path)
        full_ground_truth_path = os.path.join(root, ground_truth_path)

        # Run the Ray Tune optimization
        analysis = tune.run(
            tune.with_parameters(
                objective,
                image_path=full_image_path,
                ground_truth_path=full_ground_truth_path,
                log_wandb=log_wandb,
            ),
            config=search_space,
            metric="jaccard_cellpose",
            mode="max",
            num_samples=50,
            resume=False,
        )

        # Log the best results for the current image
        print(f"Best parameters for {image_path}: {analysis.best_config}")
        print(f"Best score for {image_path}: {analysis.best_result['jaccard_cellpose']}")

        # Close Weights & Biases for the current image
        if log_wandb:
            wandb.finish()

    ray.shutdown()


if __name__ == "__main__":
    # Run the main function
    main(data, log_wandb=False)
