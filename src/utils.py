import copy
import gc
import math
import time
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, List, Tuple, Union

import torch
import torch.distributed as dist

from model import Model


# used for benchmarks
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


def get_dummy_batch(batch_size: int, input_sentences: List[str] = None) -> List[str]:
    if input_sentences is None:
        input_sentences = copy.deepcopy(dummy_input_sentences)

    if batch_size > len(input_sentences):
        input_sentences *= math.ceil(batch_size / len(input_sentences))
    input_sentences = input_sentences[:batch_size]

    return input_sentences


def get_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--dtype", default="bfloat16", type=str)
    parser.add_argument("--max_new_tokens", default=100, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--benchmark_cycles", type=int, default=5)
    return parser


def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)
    return args


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


def benchmark_generation(model: Model, text: List[str], generate_kwargs: dict, cycles: int = 5) -> int:
    # run benchmarks for number of cycles
    total_new_tokens_generated = 0
    for _ in range(cycles):
        _, num_generated_tokens = model.generate(text, **generate_kwargs)
        total_new_tokens_generated += sum(new_tokens for new_tokens in num_generated_tokens)
    return total_new_tokens_generated


def get_benchmark_results(
    benchmark_time: float, initialization_time: float, total_new_tokens_generated: int, batch_size: int, cycles: int
) -> str:
    throughput = total_new_tokens_generated / benchmark_time
    latency = benchmark_time / cycles
    return f"""
*** Performance stats:
Throughput (including tokenization) = {throughput:.2f} tokens/sec
Throughput (including tokenization) = {1000 / throughput:.2f} msecs/token
Model loading time = {initialization_time:.2f} secs
Total tokens generated = {total_new_tokens_generated} with batch size = {batch_size}
Latency = {latency:.2f} secs
Model loading time + generation time per batch = {initialization_time + latency:.2f} secs
"""


def benchmark_end_to_end(args: Namespace, model_class: Model, text: List[str], generate_kwargs: dict) -> None:
    model, initialization_time = run_and_log_time(partial(model_class, args=args))

    print_rank_n("num params =", model.get_num_parameters())

    print_rank_n(f"generate_kwargs = {generate_kwargs}")
    print_rank_n(f"batch_size = {args.batch_size}")

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    generated_text, _ = model.generate(text, **generate_kwargs)

    for i, o in zip(text, generated_text):
        print_rank_n(f"{'-' * 60}\nINPUT = {i}\nOUTPUT = {o}\n")

    if args.benchmark_cycles > 0:
        print_rank_n(f"*** Running benchmark")

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()

        # benchmark
        total_new_tokens_generated, benchmark_time = run_and_log_time(
            partial(
                benchmark_generation,
                model=model,
                text=text,
                generate_kwargs=generate_kwargs,
                cycles=args.benchmark_cycles,
            )
        )

        print_rank_n(
            get_benchmark_results(
                benchmark_time, initialization_time, total_new_tokens_generated, args.batch_size, args.benchmark_cycles
            )
        )
