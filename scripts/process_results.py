import json
import logging
import os
import time

from cellpose_adapt.logger import setup_logging
from scripts.utils.report import generate_visual_and_quantitative_report

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import optuna
from optuna.visualization.matplotlib import plot_optimization_history

from cellpose_adapt import caching
from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.config.plotting_config import PlottingConfig
from cellpose_adapt.optimization import OptunaOptimizer
from cellpose_adapt.utils import get_device

logger = logging.getLogger(__name__)


NO_PLOTS = False
NO_REPORT = False
SHOW_PANELS = False
DEVICE = None
PLOT_ORIGINAL_IMAGE = False
EXPORT_PREDICTED_MASKS = True
CONFIGS = [
    "configs/project_configs/organoid_3d_nuclei_20231108_local.json",
    "configs/project_configs/organoid_3d_nuclei_20240220_local.json",
    "configs/project_configs/organoid_3d_nuclei_20240305_local.json",
    # add/remove entries here as you like
]


def process_project(project_config_path: str):

    if not os.path.exists(project_config_path):
        logger.error("Project config file not found at %s", project_config_path)
        return

    try:
        with open(project_config_path, 'r') as f:
            project_cfg = json.load(f)

        project_settings = project_cfg["project_settings"]
        study_name = project_settings["study_name"]
        study = optuna.create_study(study_name=study_name, storage=f"sqlite:///studies/{study_name}.db", load_if_exists=True)
        best_trial = study.best_trial

        with open(project_cfg['search_space_config_path'], 'r') as f:
            search_space_config = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to load project or search space config (%s): %s", project_config_path, e)
        return

    logger.info(f"--- Processing results for study: {study_name} ---")
    logger.info(f"Best trial #{best_trial.number} with score: {best_trial.value:.4f}")

    device = get_device(config_device=project_settings.get("device"), cli_device=DEVICE)
    cache_dir = caching.get_cache_dir(project_settings)

    # pass an empty data_pairs list here since we only need the optimizer to build a config from a trial
    optimizer = OptunaOptimizer([], search_space_config, device=device, cache_dir=cache_dir)
    best_cfg: ModelConfig = optimizer.create_config_from_trial(best_trial)

    os.makedirs("configs", exist_ok=True)
    config_filename = f"best_{study_name}_config.json"
    output_config_path = os.path.join("configs", config_filename)
    best_cfg.to_json(output_config_path)
    logger.info(f"Best configuration saved to: {output_config_path}")

    if not NO_REPORT:
        results_dir = os.path.join("reports", study_name)
        os.makedirs(results_dir, exist_ok=True)

        plotting_config = PlottingConfig()
        generate_visual_and_quantitative_report(
            cfg=best_cfg,
            project_cfg=project_cfg,
            plotting_config=plotting_config,
            results_dir=results_dir,
            device=device,
            config_filename=config_filename,
            show_panels=SHOW_PANELS,
            plot_original_image=PLOT_ORIGINAL_IMAGE,
            export_predicted_masks=EXPORT_PREDICTED_MASKS,
        )

    if not NO_PLOTS:
        plots_dir = os.path.join("reports", study_name)
        os.makedirs(plots_dir, exist_ok=True)
        logger.info(f"Generating analysis plots in: {plots_dir}")
        # Plot the metric optimization history (use 'f1_score' by convention)
        metric = 'f1_score'

        # Additionally export optimization history as a PNG using the matplotlib backend
        try:
            ax = plot_optimization_history(study, target_name=metric, error_bar=False)
            fig = ax.figure
            ax.set_title(f"{study_name.title()} Optimization History ({metric})")
            png_path = os.path.join(plots_dir, f"{study_name}_{metric}_optimization_history.png")
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization history PNG: {png_path}")
        except Exception as e:
            logger.warning(f"Could not save optimization history PNG: {e}")

    logger.info(f"--- Processing for {study_name} complete. ---")


def main():

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"processing_{timestamp}.log")
    logger = logging.getLogger(__name__)
    logger.debug("Starting script to process results from an Optuna study and generate reports (hardcoded mode).")

    for project_cfg_path in CONFIGS:
        logger.info("Processing project config: %s", project_cfg_path)
        process_project(project_cfg_path)

    logger.info("--- All processing complete. ---")


if __name__ == "__main__":
    main()
