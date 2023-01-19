import warnings
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_class", default="GPT2", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--attention_type", type=int)
    parser.add_argument("--n_positions", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_layer", type=int)

    # Runtime
    parser.add_argument("--pipeline_class", default="HF_Pipeline", type=str)
    parser.add_argument("--device", default="cuda", type=torch.device)
    parser.add_argument("--dtype", default="float16", type=lambda x: getattr(torch, x))
    parser.add_argument("--local_rank", type=int)

    # Input and output
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--max_input_length", default=-1, type=int)
    parser.add_argument("--max_new_tokens", default=100, type=int)

    # Cleanup
    parser.add_argument("--clear_every_run", action="store_true")

    # Deepspeed
    parser.add_argument("--no_inject_kernel", dest="inject_kernel", action="store_false")
    parser.add_argument("--cuda_graph", action="store_true")

    # Benchmark cycles
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=5)

    # Profiling and logging
    parser.add_argument("--max_log_outputs", default=None, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--full_trace", action="store_true")
    parser.add_argument("--show_op_names", action="store_true")
    parser.add_argument("--print_details", action="store_true")

    return parser


def check_unused(args: Namespace, defaults: Dict[str, Any], enforce=False):
    for name, default in defaults.items():
        val = getattr(args, name)
        is_default = val is None if default is None else val == default
        if not is_default:
            warnings.warn(
                f"{'Invalid' if enforce else 'Unexpected'} argument: --{name} (value ="
                f" {val}, {'setting to' if enforce else 'expected'} {default})"
            )
            if enforce:
                setattr(args, name, default)


def parse_args(argv=None, parser: ArgumentParser = None) -> Namespace:
    if parser is None:
        parser = get_arg_parser()
    args = parser.parse_args(argv)

    if args.warmup is None:
        args.warmup = args.profile

    if args.max_log_outputs is None:
        args.max_log_outputs = args.batch_size

    return args
