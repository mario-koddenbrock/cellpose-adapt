import json
import logging
import time

from cellpose_adapt import core, caching
from cellpose_adapt import io
from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.logger import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.plotting.napari_utils import show_napari
from cellpose_adapt.utils import get_device
from scripts.utils.cli import arg_parse


def main():
    args = arg_parse("Run and visualize Cellpose results for a single image.")

    # --- 1. Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"visualization_{timestamp}.log")


    # --- 2. Load Configurations ---
    cfg = ModelConfig.from_json(args.config)

    with open(args.project_config, 'r') as f:
        project_cfg = json.load(f)

    gt_mapping = project_cfg.get('gt_mapping')
    project_settings = project_cfg.get('project_settings', {})
    config_device = project_settings.get('device')
    cache_dir = caching.get_cache_dir(project_settings)
    device = get_device(cli_device=args.device, config_device=config_device)

    # --- Load Data ---
    gt_path = io.find_gt_path(args.image_path, gt_mapping) if gt_mapping else None
    image_segment, ground_truth, image = io.load_image_with_gt(args.image_path, gt_path, channel_to_segment=cfg.channel_to_segment)

    # --- 3. Initialize Model and Run Pipeline ---
    logging.info(f"Initializing model '{cfg.model_name}' on device '{device}'...")
    model = core.initialize_model(cfg.model_name, device=device)

    runner = core.CellposeRunner(model, cfg, device=device, cache_dir=cache_dir)
    masks, duration = runner.run(image_segment)

    # --- 4. Evaluate and Launch Napari Viewer ---
    metrics = {}
    if ground_truth is not None and masks is not None:
        metrics = calculate_segmentation_stats(ground_truth, masks)
        logging.info(f"Performance (full data): F1={metrics.get('f1_score', 0):.3f}, Jaccard={metrics.get('jaccard', 0):.3f}, ")

    show_napari(image, ground_truth, masks, metrics)





if __name__ == "__main__":
    main()