import copy
import logging
import logging.config
import math
import typing
from typing import Any, Callable, List, Optional, Tuple

from torch import distributed as dist


def parse_revision(pretrained_model: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    revision = None
    if pretrained_model is not None:
        pretrained_split = pretrained_model.split(":", 1)
        if len(pretrained_split) == 2:
            pretrained_model, revision = pretrained_split
    return pretrained_model, revision


def parse_config_arg(config_arg: str) -> Tuple[str, Any]:
    split_arg = [x.strip() for x in config_arg.split("=", 1)]
    if len(split_arg) != 2:
        raise ValueError(f"Cannot parse argument (not in 'key=value' format): {config_arg}")
    key, value = split_arg
    if not key.isidentifier():
        raise ValueError(f"Invalid argument (not a python identifier): {key}")
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.lower() == "none":
        value = None
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    return key, value


def parse_config_args(config_args: List[str]) -> typing.Dict[str, Any]:
    parsed_config_args = {}
    for config_arg in config_args:
        key, value = parse_config_arg(config_arg)
        if key in parsed_config_args:
            raise ValueError(f"Duplicate argument: {key}")
        parsed_config_args[key] = value
    return parsed_config_args


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


def log_dict(data: dict, logger: Callable = logging.info, rank: int = 0, _prefix=""):
    for key, value in data.items():
        if isinstance(value, dict):
            log_rank_n(f"{_prefix}{key}:", logger, rank)
            log_dict(value, logger, rank, _prefix+"  ")
        else:
            log_rank_n(f"{_prefix}{key}: {value}", logger, rank)


dummy_input_sentences = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]


def get_dummy_batch(batch_size: int, max_input_length: int = -1) -> List[str]:
    if max_input_length == -1:
        input_sentences = copy.deepcopy(dummy_input_sentences)
    else:
        input_sentences = batch_size * [" Hello" * max_input_length]

    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences
