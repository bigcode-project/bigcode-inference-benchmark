import gc
from argparse import Namespace
from functools import partial
from typing import List

import torch

from pipelines import Pipeline


def benchmark_generation(pipeline: Pipeline, text: List[str], generate_kwargs: dict, cycles: int = 5) -> int:
    # run benchmarks for number of cycles
    total_new_tokens_generated = 0
    for _ in range(cycles):
        _, num_generated_tokens = pipeline(text, **generate_kwargs)
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


def benchmark_end_to_end(args: Namespace, pipeline_class: Pipeline, text: List[str], generate_kwargs: dict) -> None:
    pipeline, initialization_time = run_and_log_time(partial(pipeline_class, args=args))

    print_rank_n("num params =", pipeline.get_num_parameters())

    print_rank_n(f"generate_kwargs = {generate_kwargs}")
    print_rank_n(f"batch_size = {args.batch_size}")

    # warmup is a must if measuring speed as it's when all the optimizations are performed
    # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
    generated_text, _ = pipeline(text, **generate_kwargs)

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
                pipeline=pipeline,
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
