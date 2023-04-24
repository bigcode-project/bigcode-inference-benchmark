import copy
import logging
import logging.config
import math
import typing
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
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
            log_dict(value, logger, rank, _prefix + "  ")
        else:
            log_rank_n(f"{_prefix}{key}: {value}", logger, rank)


dummy_inputs = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]


def get_input_lengths(batch_size, max_input_length, padding_ratio, random_state):
    """
    Generate a random set of input lengths with the desired padding ratio and at least one of the specified max length.
    """
    if padding_ratio == 0:
        return batch_size * [max_input_length]
    assert batch_size >= 2
    total_tokens = batch_size * max_input_length
    pad_tokens = round(padding_ratio * total_tokens)
    input_tokens = total_tokens - pad_tokens
    # First length is deterministic
    required_tokens = input_tokens - max_input_length
    average_length = required_tokens / (batch_size - 1)
    smin = 1
    smax = round(2 * average_length - smin)
    if smax > max_input_length:
        smax = max_input_length
        smin = round(2 * average_length - smax)
        assert smax >= smin >= 1, "Cannot obtain desired padding ratio"
    print("AA", batch_size, max_input_length, padding_ratio, smin, smax)
    assert abs(smax + smin - 2 * average_length) < 1
    for i in range(100):
        lengths = random_state.randint(smin, smax, batch_size - 2)
        remaining = required_tokens - lengths.sum()
        if 1 <= remaining <= max_input_length:
            lengths = [max_input_length, *lengths.tolist(), remaining]
            random_state.shuffle(lengths)
            assert sum(lengths) == input_tokens
            return lengths
    raise RuntimeError("Failed to get desired padding ratio")


def get_inputs_from_tokens(tokens, length, tokenizer):
    for _ in range(10):
        assert len(tokens) == length
        inputs = tokenizer.decode(tokens)
        # We often get more tokens than we started with, less in som rare cases.
        tokens = tokenizer(inputs)["input_ids"]
        if len(tokens) == length:
            return inputs
        tokens = tokens[:length] + max(length - len(tokens), 0) * [tokens[-1]]
    raise RuntimeError("Failed to generate stable input sequences")


def get_random_inputs(length, tokenizer, random_state):
    return get_inputs_from_tokens(random_state.randint(0, tokenizer.vocab_size, length).tolist(), length, tokenizer)


def get_inputs_from_files(files: List[Path], lengths, tokenizer, random_state):
    file_tokens = [tokenizer(f.open().read())["input_ids"] for f in files]
    max_len = max(len(t) for t in file_tokens)
    batch_size = len(lengths)
    inputs = []
    while len(inputs) < batch_size:
        length = lengths[len(inputs)]
        if length > max_len:
            # No file works, pick at random instead.
            inputs.append(get_random_inputs(length, tokenizer, random_state))
        else:
            tokens = file_tokens[random_state.randint(len(file_tokens))]
            if length > len(tokens):
                # Try another file.
                continue
            start_index = random_state.randint(len(tokens) - length)
            inputs.append(get_inputs_from_tokens(tokens[start_index : start_index + length], length, tokenizer))
    return inputs


def get_input_batch(
    batch_size: int,
    max_input_length: int = -1,
    tokenizer=None,
    padding_ratio: float = 0,
    seed: int = 0,
    sample_dir: Optional[Union[Path, List[Path]]] = None,
) -> List[str]:
    if max_input_length == -1:
        inputs = copy.deepcopy(dummy_inputs)
        if batch_size > len(inputs):
            inputs *= math.ceil(batch_size / len(inputs))
        return inputs[:batch_size]
    else:
        random_state = np.random.RandomState(seed)
        lengths = get_input_lengths(batch_size, max_input_length, padding_ratio, random_state)
        if isinstance(sample_dir, Path):
            if sample_dir.is_dir():
                sample_dir = [f for f in sample_dir.iterdir() if f.is_file() and f.suffix == ".py"]
            elif sample_dir.is_file():
                sample_dir = [sample_dir]
            else:
                raise FileNotFoundError(sample_dir)
        if sample_dir is None:
            return get_random_inputs(lengths, tokenizer, random_state)
        else:
            assert isinstance(sample_dir, List)
            assert len(sample_dir) > 0
            return get_inputs_from_files(sample_dir, lengths, tokenizer, random_state)
