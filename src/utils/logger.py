import logging
import os
import sys
from pathlib import Path

def get_logger(name: str = "scamshield", log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Creates or retrieves a standardized logger for ScamShield.

    Args:
        name: Name of the logger/module.
        log_file: Optional path to write log output. Defaults to 'reports/scamshield.log'.
        level: Logging severity level.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        log_dir = Path("reports")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "scamshield.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
