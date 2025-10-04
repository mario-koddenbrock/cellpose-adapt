import json
import logging
import os
import time

import optuna

from cellpose_adapt import io, caching
from cellpose_adapt.logger import get_logging_level, setup_logging
from cellpose_adapt.optimization import OptunaOptimizer
from cellpose_adapt.utils import get_device
from scripts.utils.cli import load_project_config, arg_parse

logger = logging.getLogger(__name__)
logger.debug("Starting script to run Cellpose hyperparameter optimization.")


def main():
    args = arg_parse("Run Cellpose Hyperparameter Optimization.")

    project_cfg = load_project_config(args.project_config)
    if not project_cfg:
        return

    project_settings = project_cfg['project_settings']

    # --- Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_level = get_logging_level(project_settings.get("logging_level", logging.INFO))
    setup_logging(log_level=log_level, log_file=f"optimization_{timestamp}.log")

    # Extract project_settings from the config
    project_settings = project_cfg["project_settings"]
    study_name = project_settings["study_name"]
    device = get_device(project_settings.get("device"))
    n_trials = project_settings["n_trials"]
    limit_per_source = project_settings.get("limit_images_per_source")
    cache_dir = caching.get_cache_dir(project_settings)
    iou_threshold = project_settings.get("iou_threshold")

    data_sources = project_cfg["data_sources"]
    gt_mapping = project_cfg["gt_mapping"]
    search_config_path = project_cfg["search_space_config_path"]

    os.makedirs("studies", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # --- Load Data and Search Space ---
    # Pass the limit to the find_image_gt_pairs function
    data_pairs = io.find_image_gt_pairs(data_sources, gt_mapping, limit_per_source)
    if not data_pairs:
        logging.error("No data pairs found based on the provided configuration. Exiting.")
        return

    try:
        with open(search_config_path, "r") as f:
            search_space_config = json.load(f)
        logging.info("Loaded search space from %s", search_config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error("Failed to load or parse search config file %s: %s", search_config_path, e)
        return

    # --- Initialize and Run Optimizer ---
    optimizer = OptunaOptimizer(
        data_pairs,
        search_space_config,
        device=device,
        cache_dir=cache_dir,
        iou_threshold=iou_threshold,
    )

    storage_url = f"sqlite:///studies/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler = optuna.samplers.TPESampler(seed=42),
    )

    logging.info(
        "Starting/resuming optimization for study '%s' with %d trials.",
        study_name,
        n_trials,
    )

    study.optimize(optimizer.objective, n_trials=n_trials, n_jobs=1)

    # --- Log Final Results ---
    logging.info("Optimization finished.")
    logging.info("Best trial for study '%s':", study.study_name)
    logging.info("  Value (Mean Jaccard): %.4f", study.best_value)
    logging.info("  Params: ")
    for key, value in study.best_params.items():
        logging.info("    %s: %s", key, value)


if __name__ == "__main__":
    main()
