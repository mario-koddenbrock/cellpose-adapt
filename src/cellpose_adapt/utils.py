import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)
logger.debug("Utility module loaded. Version: %s", torch.__version__)


def get_device(config_device: str = None, cli_device: str = None) -> torch.device:
    """
    Determines the torch device based on priority: CLI > Config > Auto-detect.
    """
    if cli_device:
        logging.info(f"Using device specified on command line: '{cli_device}'")
        return torch.device(cli_device)

    if config_device:
        logging.info(f"Using device specified in project config: '{config_device}'")
        return torch.device(config_device)

    # Auto-detection fallback
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"No device specified, auto-detected: '{device}'")
    return device


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = f"{seed}"
