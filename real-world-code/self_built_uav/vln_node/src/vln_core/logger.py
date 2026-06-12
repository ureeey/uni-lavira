# vln_node/src/vln_core/logger.py
import logging
import sys

def _build_logger(name: str = "vln_node") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.propagate = False
    return logger

logger = _build_logger()
