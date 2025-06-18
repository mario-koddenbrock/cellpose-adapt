import argparse
import logging
import os

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)

from cellpose_adapt.config import PipelineConfig
from cellpose_adapt.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Analyze an Optuna study and save the best config."
    )
    parser.add_argument(
        "--study_db",
        type=str,
        required=True,
        help="Path to the Optuna study SQLite DB.",
    )
    parser.add_argument(
        "--output_config",
        type=str,
        required=True,
        help="Path to save the best configuration JSON file.",
    )
    args = parser.parse_args()

    setup_logging(log_file="analysis.log")

    if not os.path.exists(args.study_db):
        logging.error("Study database not found at %s", args.study_db)
        return

    study = optuna.load_study(study_name=None, storage=f"sqlite:///{args.study_db}")

    logging.info("Study analysis:")
    logging.info("  Number of finished trials: %d", len(study.trials))

    best_trial = study.best_trial
    logging.info("  Best trial number: %d", best_trial.number)
    logging.info("  Best value (Jaccard): %.4f", best_trial.value)
    logging.info("  Best parameters:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")

    # Create a full config object from the best parameters
    # Get a default config to fill in non-optimized parameters
    best_config = PipelineConfig()
    for key, value in best_trial.params.items():
        if hasattr(best_config, key):
            setattr(best_config, key, value)

    # Save the best configuration
    output_dir = os.path.dirname(args.output_config)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    best_config.to_json(args.output_config)
    logging.info("Best configuration saved to %s", args.output_config)

    # Generate and save visualization plots
    study_dir = os.path.dirname(args.study_db)

    try:
        fig_history = plot_optimization_history(study)
        history_path = os.path.join(study_dir, "optimization_history.html")
        fig_history.write_html(history_path)
        logging.info("Saved optimization history plot to %s", history_path)

        fig_importance = plot_param_importances(study)
        importance_path = os.path.join(study_dir, "param_importances.html")
        fig_importance.write_html(importance_path)
        logging.info("Saved parameter importance plot to %s", importance_path)

        fig_slice = plot_slice(study)
        slice_path = os.path.join(study_dir, "slice_plot.html")
        fig_slice.write_html(slice_path)
        logging.info("Saved slice plot to %s", slice_path)

    except (ValueError, ImportError) as e:
        logging.warning(
            "Could not generate plots. This may happen with few trials or if plotly is not fully installed. Error: %s",
            e,
        )


if __name__ == "__main__":
    main()
