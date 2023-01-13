import contextlib
import gc
from argparse import Namespace
from functools import partial
from typing import List, Tuple, Type

import numpy as np
import torch

from src.pipelines.pipeline import Pipeline

from src.utils.utils import print_rank_n, run_and_log_time, format_ms


def benchmark_generation(pipeline: Pipeline, text: List[str], generate_kwargs: dict, cycles: int = 5) -> Tuple[int,np.ndarray]:
    # run benchmarks for number of cycles
    total_new_tokens_generated = 0
    benchmark_times=[]
    for _ in range(cycles):
        _, num_generated_tokens, times = pipeline(text, **generate_kwargs)
        total_new_tokens_generated += sum(new_tokens for new_tokens in num_generated_tokens)
        benchmark_times.append(times)
    return total_new_tokens_generated, np.array(benchmark_times)


def print_benchmark_results(
    benchmark_times: np.ndarray, initialization_time: float, total_new_tokens_generated: int, batch_size: int
):
    cycles = benchmark_times.shape[0]
    assert benchmark_times.shape[1] == 3

    step_time=benchmark_times.mean(axis=0)
    batch_times=np.array(benchmark_times).sum(axis=1)
    average_time=batch_times.mean()

    throughput = total_new_tokens_generated / cycles / average_time
    model_throughput = total_new_tokens_generated / cycles / step_time[1]
    print_rank_n("*** Performance stats:")
    print_rank_n(f"Model loading time = {format_ms(initialization_time)}")

    print_rank_n(f"Latency (end to end) = {format_ms(average_time)}")
    print_rank_n(f"Latency (tokenization) = {format_ms(step_time[0])}")
    print_rank_n(f"Latency (model) = {format_ms(step_time[1])}")
    print_rank_n(f"Latency (decode) = {format_ms(step_time[2])}")
    print_rank_n(f"Latency (max) = {format_ms(batch_times.max())}")
    print_rank_n(f"Latency (min) = {format_ms(batch_times.min())}")
    print_rank_n(f"Model loading time + generation time per batch = {format_ms(initialization_time + average_time)}")

    print_rank_n(f"Throughput (model) = {model_throughput:.2f} tokens/s")
    print_rank_n(f"Throughput (end to end) = {throughput:.2f} tokens/s")
    print_rank_n(f"Token time (end to end) = {format_ms(throughput**-1)}/token")
    print_rank_n(f"Total tokens generated = {total_new_tokens_generated} with batch size = {batch_size}")

def get_trace_fn(args):
    def trace_fn(p):
        averages=p.key_averages()
        if args.full_trace:
            # Show every op
            print(p.profiler.table(row_limit=-1, max_src_column_width=1000))
        if args.print_op_names:
            # Show non-cropped names
            for entry in averages:
                print(entry.key)

        # Try to avoid name cropping, still hard-coded to max 55 characters
        print(averages.table(sort_by="self_cuda_time_total", row_limit=-1, max_src_column_width=1000))

    return trace_fn


def benchmark_end_to_end(args: Namespace, pipeline_class: Type[Pipeline], text: List[str], generate_kwargs: dict) -> None:
    pipeline: Pipeline
    pipeline, initialization_time = run_and_log_time(partial(pipeline_class, args=args))

    print_rank_n("num params =", pipeline.get_num_parameters())

    print_rank_n(f"generate_kwargs = {generate_kwargs}")
    print_rank_n(f"batch_size = {args.batch_size}")

    warmup = args.warmup
    if warmup is None:
        warmup=args.profile

    total_new_tokens_generated = 0
    benchmark_times=[]

    if args.profile:
        schedule = torch.profiler.schedule(
            # Warmup is a must if measuring speed as it's when all the optimizations are performed
            # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
            skip_first = args.skip,
            # Warmup for the profiler
            warmup = warmup,
            wait=0,
            active=args.cycles,
        )
        p = torch.profiler.profile(
            schedule=schedule,
            activities=[torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=get_trace_fn(args)
        )
    else:
        p = contextlib.nullcontext()

    with p:
        for step in range(args.skip+warmup+args.cycles):
            generated_text, num_generated_tokens, times = pipeline(text, **generate_kwargs)
            if args.profile:
                p.step()

            if step == 0:
                for i, o, _ in zip(text, generated_text, range(args.max_log_outputs)):
                    print_rank_n(f"{'-' * 60}\nINPUT = {i}\nOUTPUT = {o}\n")

            if step >= args.skip+warmup:
                total_new_tokens_generated += sum(new_tokens for new_tokens in num_generated_tokens)
                benchmark_times.append(times)

            if args.clear_every_run:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

    if total_new_tokens_generated > 0:
        print_benchmark_results(
            np.array(benchmark_times), initialization_time, total_new_tokens_generated, args.batch_size
        )
