import argparse
import json
import logging
import os
import time

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.config.plotting_config import PlottingConfig
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.optimization import OptunaOptimizer
from cellpose_adapt.utils import get_device
from reporting_utils import generate_visual_and_quantitative_report

logger = logging.getLogger(__name__)
logger.debug("Starting script to process results from an Optuna study and generate reports.")


def main():
    parser = argparse.ArgumentParser(description="Analyze a study and generate all reports.")
    parser.add_argument("--project_config", type=str, help="Path to the original project JSON config file.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating HTML analysis plots.")
    parser.add_argument("--no-report", action="store_true", help="Skip generating visual/quantitative reports.")
    parser.add_argument("--show-panels", action="store_true", help="Display generated report panels on screen.")
    parser.add_argument("--device", type=str, default=None, help="Override device setting ('cpu', 'cuda', 'mps').")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"processing_{timestamp}.log")

    plotting_config = PlottingConfig()



    if not os.path.exists(args.project_config):
        logging.error("Project config file not found at %s", args.project_config)
        return


    try:

        with open(args.project_config, 'r') as f:
            project_cfg = json.load(f)

        project_settings = project_cfg["project_settings"]
        study_name = project_settings["study_name"]
        study = optuna.create_study(study_name=study_name, storage=f"sqlite:///studies/{study_name}.db", load_if_exists=True)
        best_trial = study.best_trial

        with open(project_cfg['search_space_config_path'], 'r') as f:
            search_space_config = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to load project or search space config: {e}")
        return

    logging.info(f"--- Processing results for study: {study_name} ---")
    logging.info(f"Best trial #{best_trial.number} with score: {best_trial.value:.4f}")

    # --- 2. Create and Save Best Config ---
    device = get_device(cli_device=args.device, config_device=project_settings.get("device"))
    cache_dir = project_settings.get("cache_dir", ".cache")
    optimizer = OptunaOptimizer(None, search_space_config, device=device, cache_dir =cache_dir)
    best_cfg:ModelConfig = optimizer.create_config_from_trial(best_trial)
    # best_cfg = ModelConfig.from_json("configs/manual_organoid_3d_nuclei_study_config.json")


    os.makedirs("configs", exist_ok=True)
    config_filename = f"best_{study_name}_config.json"
    output_config_path = os.path.join("configs", config_filename)
    best_cfg.to_json(output_config_path)
    logging.info(f"Best configuration saved to: {output_config_path}")

    # --- 3. Generate Visual and Quantitative Reports ---
    if not args.no_report:
        results_dir = os.path.join("reports", study_name)
        os.makedirs(results_dir, exist_ok=True)

        generate_visual_and_quantitative_report(
            cfg=best_cfg,
            project_cfg=project_cfg,
            plotting_config=plotting_config,
            results_dir=results_dir,
            device=device,
            config_filename=config_filename,
            show_panels=args.show_panels
        )

    # --- 4. Generate Analysis Plots ---
    if not args.no_plots:
        plots_dir = os.path.join("reports", study_name)
        os.makedirs(plots_dir, exist_ok=True)
        logging.info(f"Generating analysis plots in: {plots_dir}")
        try:
            plot_optimization_history(study).write_html(os.path.join(plots_dir, "optimization_history.html"))
            plot_param_importances(study).write_html(os.path.join(plots_dir, "param_importances.html"))
            plot_slice(study).write_html(os.path.join(plots_dir, "slice_plot.html"))
            logging.info("Analysis plots saved successfully.")
        except (ValueError, ImportError) as e:
            logging.warning(f"Could not generate plots: {e}")

    logging.info("--- Processing complete. ---")


if __name__ == "__main__":
    main()