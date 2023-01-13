import logging
import logging.config
from typing import Callable

from torch import distributed as dist


def configure_logging(name=None):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": f"%(asctime)s{'' if name is None else ' ['+name+']'}: %(message)s",
                "use_colors": True,
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {"default": {"level": "DEBUG", "handlers": ["default"]}},
        "root": {"handlers": ["default"], "level": "INFO"},
    }
    logging.config.dictConfig(logging_config)


def log_rank_n(msg: str, logger: Callable = logging.info, rank: int = 0):
    if rank < 0 or not dist.is_initialized() or dist.get_rank() == rank:
        # Multi-line logs break formatting
        for line in msg.splitlines():
            logger(line)


def log_dict(data: dict, logger: Callable = logging.info, rank: int = 0):
    for key, value in data.items():
        log_rank_n(f"{key}: {value}", logger, rank)


def format_ms(t: float):
    return f"{1000 * t:.2f} ms"
