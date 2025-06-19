import argparse
import json
import logging
import os
import time

import optuna
import torch

from cellpose_adapt import io
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.optimization import OptunaOptimizer


def get_device(device_str: str = None) -> torch.device:
    """Determines the torch device, either from config or by auto-detection."""
    if device_str:
        logging.info(f"Using device specified in config: '{device_str}'")
        return torch.device(device_str)

    # Auto-detection fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"No device specified, auto-detected: '{device}'")
    return device


def get_logging_level(level_str: str = "INFO") -> int:
    """Maps a string to a logging level constant."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(level_str.upper(), logging.INFO)
    if level_str.upper() not in level_map:
        logging.warning(f"Invalid logging level '{level_str}'. Defaulting to 'INFO'.")
    return level


def load_project_config(config_path: str) -> dict:
    """Loads and validates the main project configuration file."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.info("Loaded project configuration from %s", config_path)

        # Basic validation
        if (
            "project_settings" not in config
            or "search_space_config_path" not in config
            or "data_sources" not in config
            or "gt_mapping" not in config
        ):
            raise KeyError(
                "Config must contain 'project_settings', 'search_space_config_path', 'data_sources' and 'gt_mapping' keys."
            )

        return config
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(
            "Failed to load or validate project config file %s: %s", config_path, e
        )
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Cellpose Hyperparameter Optimization.")
    parser.add_argument("project_config_path", type=str, help="Path to the JSON project config.")
    args = parser.parse_args()

    project_config = load_project_config(args.project_config_path)
    if not project_config:
        return

    settings = project_config['project_settings']

    # --- Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_level = get_logging_level(settings.get("logging_level", "INFO"))
    setup_logging(log_level=log_level, log_file=f"optimization_{timestamp}.log")


    # Extract settings from the config
    settings = project_config["project_settings"]
    study_name = settings["study_name"]
    device = get_device(settings.get("device"))
    n_trials = settings["n_trials"]
    limit_per_source = settings.get("limit_images_per_source")

    data_sources = project_config["data_sources"]
    gt_mapping = project_config["gt_mapping"]
    search_config_path = project_config["search_space_config_path"]

    os.makedirs("studies", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # --- Load Data and Search Space ---
    # Pass the limit to the find_image_gt_pairs function
    data_pairs = io.find_image_gt_pairs(data_sources, gt_mapping, limit_per_source)
    if not data_pairs:
        logging.error(
            "No data pairs found based on the provided configuration. Exiting."
        )
        return

    try:
        with open(search_config_path, "r") as f:
            search_space_config = json.load(f)
        logging.info("Loaded search space from %s", search_config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(
            "Failed to load or parse search config file %s: %s", search_config_path, e
        )
        return

    # --- Initialize and Run Optimizer ---
    optimizer = OptunaOptimizer(data_pairs, search_space_config, device=device)

    storage_url = f"sqlite:///studies/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
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
