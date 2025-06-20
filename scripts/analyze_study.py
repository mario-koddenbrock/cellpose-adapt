import argparse
import logging
import os
import time

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

from cellpose_adapt.config import PipelineConfig
from cellpose_adapt.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Analyze an Optuna study and save the best config.")
    parser.add_argument("--study_db", type=str, required=True, help="Path to the Optuna study SQLite DB.")
    parser.add_argument("--output_config", type=str, required=True, help="Path to save the best configuration JSON file.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"analysis_{timestamp}.log")


    if not os.path.exists(args.study_db):
        logging.error("Study database not found at %s", args.study_db)
        return

    study = optuna.load_study(study_name=None, storage=f"sqlite:///{args.study_db}")

    logging.info("Study analysis:")
    logging.info("  Number of finished trials: %d", len(study.trials))

    best_trial = study.best_trial
    logging.info("  Best trial number: %d", best_trial.number)
    logging.info("  Best value (F1): %.4f", best_trial.value)
    logging.info("  Best parameters:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # --- Save the Best Configuration ---
    # Create a full config object from the best parameters
    best_config = PipelineConfig()
    # Combine fixed params (if any) and best trial params for a complete config
    # Note: This assumes fixed_params are not in the study, which is correct.
    # To be fully robust, one might load the original search config, but this is fine.
    for key, value in best_trial.params.items():
        if hasattr(best_config, key):
            setattr(best_config, key, value)

    output_dir = os.path.dirname(args.output_config)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    best_config.to_json(args.output_config)
    logging.info("Best configuration saved to %s", args.output_config)

    # --- Generate and Save Visualization Plots to a Specific Report Folder ---

    # **MODIFIED LOGIC STARTS HERE**
    # 1. Define the new output directory for plots
    study_name = study.study_name
    output_plot_dir = os.path.join("reports", f"{study_name}")

    # 2. Create the directory
    os.makedirs(output_plot_dir, exist_ok=True)
    logging.info("Saving analysis plots to: %s", output_plot_dir)

    try:
        # 3. Save plots to the new directory
        fig_history = plot_optimization_history(study)
        history_path = os.path.join(output_plot_dir, "optimization_history.html")
        fig_history.write_html(history_path)
        logging.info("  - Saved optimization history plot.")

        fig_importance = plot_param_importances(study)
        importance_path = os.path.join(output_plot_dir, "param_importances.html")
        fig_importance.write_html(importance_path)
        logging.info("  - Saved parameter importance plot.")

        fig_slice = plot_slice(study)
        slice_path = os.path.join(output_plot_dir, "slice_plot.html")
        fig_slice.write_html(slice_path)
        logging.info("  - Saved slice plot.")

    except (ValueError, ImportError) as e:
        logging.warning("Could not generate plots. This may happen with few trials or if plotly is not fully installed. Error: %s", e)

if __name__ == "__main__":
    main()