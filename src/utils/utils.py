import logging
import logging.config
import sys
import time
from functools import partial
from typing import Any, List, Tuple, Union

import torch.distributed as dist


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

    # Add these methods so that stdout can be redirected to logging.
    logging.write = lambda msg: logging.info(msg) if msg != '\n' else None
    logging.flush = lambda : None

    sys.stdout=logging
    sys.stderr=logging



def run_and_log_time(execs: Union[List[partial], partial]) -> Tuple[Union[List[Any], Any], float]:
    # runs a function / list of functions and times them
    start_time = time.time()

    if type(execs) == list:
        results = []
        for f in execs:
            results.append(f())
    else:
        results = execs()

    time_elapsed = time.time() - start_time
    return results, time_elapsed


def run_rank_n(func: partial, barrier: bool = False, rank: int = 0, other_rank_output: Any = None) -> Any:
    # runs function on only process with specified rank
    if dist.is_initialized():
        if dist.get_rank() == rank:
            output = func()
            if barrier:
                dist.barrier()
            return output
        else:
            if barrier:
                dist.barrier()
            return other_rank_output
    else:
        return func()


def print_rank_n(*values, rank: int = 0) -> None:
    # print on only process with specified rank
    if dist.is_initialized():
        if dist.get_rank() == rank:
            print(*values)
    else:
        print(*values)

def format_ms(t: float):
    return f"{1000 * t:.2f} ms"