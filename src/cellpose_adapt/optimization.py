import logging

import numpy as np
import optuna
import torch
from tqdm import tqdm

from . import core, io
from .config.model_config import ModelConfig
from .metrics import calculate_segmentation_stats

logger = logging.getLogger(__name__)
logger.debug("Optuna optimizer module loaded. Version: %s", optuna.__version__)

class OptunaOptimizer:
    """Manages the Optuna hyperparameter optimization process."""

    def __init__(self,
                 data_pairs: list,
                 search_space_config: dict,
                 device: torch.device,
                 iou_threshold:float = 0.5,
                 cache_dir: str = ".cache",
                 ):
        self.data_pairs = data_pairs
        self.search_space_config = search_space_config
        self.device = device
        self.cache_dir = cache_dir
        self.iou_threshold = iou_threshold

        self.fixed_params = self.search_space_config.get("fixed_params", {})
        self.search_space = self.search_space_config.get("search_space", {})
        self.loaded_models = {}

    def _get_model(self, model_name: str):
        """Loads a model if not already loaded, using the configured device."""
        if model_name not in self.loaded_models:
            self.loaded_models[model_name] = core.initialize_model(model_name, device=self.device)
        return self.loaded_models[model_name]

    def create_config_from_trial(self, trial: optuna.Trial) -> ModelConfig:
        """Builds a ModelConfig from a combination of fixed and suggested parameters."""

        # Start with fixed parameters
        params = self.fixed_params.copy()

        # Add parameters suggested by Optuna
        for name, suggestion in self.search_space.items():
            s_type = suggestion[0]
            s_args = suggestion[1:]

            if s_type == "suggest_categorical":
                params[name] = trial.suggest_categorical(name, *s_args)
            elif s_type == "suggest_int":
                # Check for log scale flag
                use_log = "log" in s_args
                if use_log:
                    s_args.remove("log")
                params[name] = trial.suggest_int(name, *s_args, log=use_log)
            elif s_type == "suggest_float":
                # Check for log scale flag
                use_log = "log" in s_args
                if use_log:
                    s_args.remove("log")
                params[name] = trial.suggest_float(name, *s_args, log=use_log)
            else:
                raise ValueError(
                    f"Unknown suggestion type '{s_type}' for parameter '{name}'"
                )

        return ModelConfig(**params)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function. Creates a config, runs it on the dataset,
        and returns the mean Jaccard score.
        """
        try:
            trial_cfg = self.create_config_from_trial(trial)
        except Exception as e:
            logger.error("Error creating config from trial: %s", e)
            raise optuna.exceptions.TrialPruned(f"Config creation failed: {e}")

        # Get the appropriate model, loading it only if necessary
        model = self._get_model(trial_cfg.model_name)

        # Create a lightweight runner with the preloaded model
        runner = core.CellposeRunner(model, trial_cfg, device=self.device, cache_dir=self.cache_dir)
        channel_to_segment = trial_cfg.channel_to_segment
        
        scores = []
        pbar = tqdm(self.data_pairs, desc=f"Trial {trial.number}", leave=False)
        for image_path, gt_path in pbar:
            image, ground_truth, _ = io.load_image_with_gt(image_path, gt_path, channel_to_segment)
            if image is None or ground_truth is None:
                continue

            masks, _ = runner.run(image)
            if masks is None:
                scores.append(0.0)  # Penalize failures
                continue

            metrics = calculate_segmentation_stats(ground_truth, masks, iou_threshold=self.iou_threshold)
            score = metrics["f1_score"]
            scores.append(score)
            pbar.set_postfix({"last_score": f"{score:.3f}"})

        if not scores:
            logger.warning("No images could be processed in this trial. Pruning.")
            raise optuna.exceptions.TrialPruned()

        mean_score = float(np.mean(scores))
        trial.set_user_attr("mean_score", mean_score)
        logger.info("Trial %d: %.4f", trial.number, mean_score)
        try:
            logger.info(f"Best: {trial.study.best_value:.4f} (Trial {trial.study.best_trial.number})")
        except ValueError:
            logger.info("No best trial found yet.")
        return mean_score
