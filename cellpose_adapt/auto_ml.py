import os
import optuna
from cellpose_adapt.config import CellposeConfig
from cellpose_adapt.core import evaluate_model
from cellpose_adapt.file_io import load_image_with_gt
from cellpose_adapt.utils import set_all_seeds
from experiments.data import data

set_all_seeds(42)


# Objective function for Optuna
def objective(trial, image_path, ground_truth_path):
    key = os.path.basename(image_path).replace(".tif", "")

    image_name = os.path.basename(image_path).replace(".tif", "")
    image_orig, ground_truth = load_image_with_gt(image_path, ground_truth_path)

    if image_orig is None or ground_truth is None:
        raise optuna.TrialPruned("Invalid image or ground truth.")

    if "nuclei" in ground_truth_path.lower():
        params_type = "Nuclei"
        channel_idx = 0
    elif "membrane" in ground_truth_path.lower():
        params_type = "Membranes"
        channel_idx = 1
    else:
        raise ValueError(f"Invalid ground truth: {ground_truth_path}")

    # Define the hyperparameter search space
    params = {
        "model_name": trial.suggest_categorical("model_name", [
            "cyto3", "cyto2_cp3", "cyto", "cyto2", "nuclei",
            "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3",
            "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3",
            "deepbacs_cp3"]),
        "diameter": trial.suggest_loguniform("diameter", 1, 100000),
        "invert": trial.suggest_categorical("invert", [True, False]),
        "norm3D": trial.suggest_categorical("norm3D", [True, False]),
        "normalize": trial.suggest_categorical("normalize", [True, False]),
        "percentile_min": trial.suggest_uniform("percentile_min", 0.0, 10.0),
        "percentile_max": trial.suggest_uniform("percentile_max", 90.0, 100.0),
        "sharpen_radius": trial.suggest_uniform("sharpen_radius", 0, 500),
        "smooth_radius": trial.suggest_uniform("smooth_radius", 0, 500),
        "tile_norm_blocksize": 0, # trial.suggest_uniform("tile_norm_blocksize", 0, 500),
        "tile_norm_smooth3D": 0, # trial.suggest_uniform("tile_norm_smooth3D", 0, 500),
        "cellprob_threshold": 0, # trial.suggest_uniform("cellprob_threshold", -30, 30),
        "channel_axis": None,
        "channel_segment": 0,
        "channel_nuclei": 0,
        "do_3D": trial.suggest_categorical("do_3D", [True, False]),
        "flow_threshold": trial.suggest_uniform("flow_threshold", 0, 1),
        "interp": trial.suggest_categorical("interp", [True, False]),
        "max_size_fraction": trial.suggest_uniform("max_size_fraction", 0.1, 1.0),
        "min_size": trial.suggest_loguniform("min_size", 1, 100000),
        "niter": trial.suggest_int("niter", 50, 500),
        "tile_overlap": trial.suggest_uniform("tile_overlap", 0.0, 0.5),
        "stitch_threshold": trial.suggest_uniform("stitch_threshold", 0.0, 1.0),
    }

    config = CellposeConfig(type=params_type, **params)

    # Get the appropriate channel
    image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig

    try:
        # Evaluate the model
        results = evaluate_model(
            key, image, ground_truth, config,
            cache_dir=".cache", separate_mask_computing=True,
            only_cached_results=False,
        )

        if not isinstance(results, dict) or "jaccard_cellpose" not in results:
            raise optuna.TrialPruned("No valid results for Jaccard metric.")

        # Return the objective value (maximize Jaccard score)
        return results["jaccard_cellpose"]

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise optuna.TrialPruned(f"Exception encountered: {e}")


def main(data, root="../"):
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

    root = os.path.abspath(root)

    for image_idx, (image_path, ground_truth_path) in enumerate(data):
        print(f"Optimizing parameters for image {image_idx + 1}/{len(data)}: {image_path}")

        full_image_path = os.path.join(root, image_path)
        full_ground_truth_path = os.path.join(root, ground_truth_path)

        # Create a study for optimization
        study_name = f"cellpose_{image_idx}_{os.path.basename(image_path)}"
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(
            lambda trial: objective(trial, full_image_path, full_ground_truth_path),
            n_trials=100,
        )

        # Log the best result for the current image
        print(f"Best parameters for {image_path}: {study.best_params}")
        print(f"Best Jaccard score for {image_path}: {study.best_value}")


if __name__ == "__main__":
    main(data)
