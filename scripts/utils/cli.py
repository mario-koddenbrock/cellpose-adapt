import argparse
import json
import logging


def arg_parse(description):
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the final pipeline configuration JSON file (e.g., best_cfg.json)."
    )
    parser.add_argument(
        "--project_config",
        type=str,
        required=True,
        help="Path to the project config file to find GT mapping and device settings."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device setting ('cpu', 'cuda', 'mps')."
    )


    args = parser.parse_args()
    return args

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