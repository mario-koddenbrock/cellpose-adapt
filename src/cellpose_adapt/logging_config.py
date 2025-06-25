import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DIR = "logs"


def setup_logging(log_level=logging.INFO, log_file="app.log"):
    """
    Configures logging to both console and a rotating file.
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    log_path = os.path.join(LOG_DIR, log_file)

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stdout_handler)

    # File Handler (rotates daily)
    file_handler = TimedRotatingFileHandler(log_path, when="midnight", backupCount=5)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    logging.info("Logging configured to console and %s", log_path)


def get_logging_level(level_str: str = "INFO") -> int:
    """Maps a string to a logging level constant."""
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level_map.get(level_str.upper(), logging.INFO)
    if level_str.upper() not in level_map:
        logging.warning(f"Invalid logging level '{level_str}'. Defaulting to 'INFO'.")
    return level
