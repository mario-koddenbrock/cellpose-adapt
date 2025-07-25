import argparse


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Run and visualize Cellpose results for a single image."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--config_path",
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