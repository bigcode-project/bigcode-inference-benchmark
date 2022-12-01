from argparse import ArgumentParser, Namespace

import torch


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--pipeline_class", default="HF_GPU_Pipeline", type=str)
    parser.add_argument("--model_class", default="GPT2", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--dtype", default="bfloat16", type=str)
    parser.add_argument("--max_new_tokens", default=100, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--n_positions", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--benchmark_cycles", type=int, default=5)
    return parser


def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    return args
