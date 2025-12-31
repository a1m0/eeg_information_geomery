import logging


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger("deap_ig_revised")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
