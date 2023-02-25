import typing
import warnings
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_type")
    parser.add_argument("--pretrained_config")
    parser.add_argument("--pretrained_model")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("config_args", nargs="*")

    # Runtime
    parser.add_argument("--pipeline_class", default="HF_Pipeline")
    parser.add_argument("--device", default="cuda", type=torch.device)
    parser.add_argument("--dtype", default="float16", type=lambda x: getattr(torch, x))
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--no_fast_init", dest="fast_init", action="store_false")

    # Input and output
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_input_length", default=-1, type=int)
    parser.add_argument("--max_new_tokens", default=100, type=int)

    # Cleanup
    parser.add_argument("--clear_every_run", action="store_true")

    # Benchmark cycles
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=5)

    # Profiling and logging
    parser.add_argument("--max_log_outputs", default=None, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--full_trace", action="store_true")
    parser.add_argument("--show_op_names", action="store_true")

    return parser


def parse_config_args(config_args: List[str]) -> typing.Dict[str, Any]:
    parsed_config_args = {}
    for config_arg in config_args:
        split_arg = [x.strip() for x in config_arg.split("=", 1)]
        if len(split_arg) != 2:
            raise ValueError(f"Cannot parse argument (not in 'key=value' format): {config_arg}")
        key, value = split_arg
        if not key.isidentifier():
            raise ValueError(f"Invalid argument (not a python identifier): {key}")
        if key in parsed_config_args:
            raise ValueError(f"Duplicate argument: {key}")
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
        parsed_config_args[key] = value
    return parsed_config_args


def parse_args(argv=None, parser: ArgumentParser = None) -> Namespace:
    if parser is None:
        parser = get_arg_parser()
    args = parser.parse_args(argv)
    args.config_args = parse_config_args(args.config_args)

    return args
