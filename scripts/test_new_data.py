import json
import logging
import os
import time

from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.config.plotting_config import PlottingConfig
from cellpose_adapt.logger import setup_logging
from cellpose_adapt.utils import get_device
from scripts.utils.cli import arg_parse
from scripts.utils.report import generate_visual_and_quantitative_report

logger = logging.getLogger(__name__)
logger.debug("Starting script to test given config on new data.")


def main():
    args = arg_parse("Test a configuration on a dataset and generate a report.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"testing_{timestamp}.log")

    plotting_config = PlottingConfig()

    # --- 1. Load Configs ---
    try:
        cfg = ModelConfig.from_json(args.config)
        with open(args.project_config, 'r') as f:
            project_cfg = json.load(f)
        project_settings = project_cfg["project_settings"]
        study_name = project_settings.get('study_name')
        if not study_name:
            # Fallback to a name based on the config file if study_name is not in project settings
            study_name = os.path.splitext(os.path.basename(args.config))[0]
            logging.warning(f"No 'study_name' in project config, using fallback name: {study_name}")

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to load a required config file: {e}")
        return

    logging.info(f"--- Testing config for project: {study_name} ---")

    # --- 2. Generate Visual and Quantitative Reports ---
    no_report = False
    show_panels = True
    if not no_report:
        # Use the non-timestamped directory name as you specified
        results_dir = os.path.join("reports", f"test_{study_name}")
        os.makedirs(results_dir, exist_ok=True)

        device = get_device(cli_device=args.device, config_device=project_settings.get("device"))

        generate_visual_and_quantitative_report(
            cfg=cfg,
            project_cfg=project_cfg,
            plotting_config=plotting_config,
            results_dir=results_dir,
            device=device,
            config_filename="tested_config.json",
            show_panels=show_panels
        )

    logging.info("--- Testing complete. ---")


if __name__ == "__main__":
    main()