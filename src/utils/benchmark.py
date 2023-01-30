import contextlib
import gc
import logging
from argparse import Namespace
from functools import partial
from typing import Any, Dict, List, Type, Union

import numpy as np
import torch

from src.constants import DECODE_TIME, END_TO_END_TIME, METRIC_KEYS, MODEL_TIME, NUM_GENERATED_TOKENS, TOKENIZE_TIME
from src.pipelines.pipeline import Pipeline
from src.utils.logging import format_ms, log_dict, log_rank_n
from src.utils.utils import run_and_log_time


logger = logging.getLogger(__name__)


def get_trace_fn(args, rank=-1):
    def trace_fn(
        p: torch.profiler.profile,
    ):
        averages = p.key_averages()
        if args.full_trace:
            # Show every GPU op.
            # Exclude CPU cuda ops to shorten the table.
            events = torch.autograd.profiler.EventList(
                [evt for evt in p.profiler.function_events if evt.self_cuda_time_total > 0]
            )
            log_rank_n(events.table(row_limit=-1, max_src_column_width=1000), logger.info, rank)

        if args.show_op_names:
            # Show non-cropped names, in the same order as in the table.
            averages_sorted = torch.autograd.profiler.EventList(
                sorted(averages, key=lambda evt: evt.self_cuda_time_total, reverse=True)
            )
            for entry in averages_sorted:
                log_rank_n(entry.key, logger.info, rank)

        # Try to avoid name cropping, still hard-coded to max 55 characters
        log_rank_n(
            averages.table(sort_by="self_cuda_time_total", row_limit=-1, max_src_column_width=1000), logger.info, rank
        )

    return trace_fn


def get_profiler(args: Namespace) -> Union[torch.profiler.profile, contextlib.nullcontext]:
    schedule = torch.profiler.schedule(
        # Warmup is a must if measuring speed as it's when all the optimizations are performed
        # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
        skip_first=args.skip,
        # Warmup for the profiler
        warmup=args.warmup,
        wait=0,
        active=args.cycles,
    )
    return torch.profiler.profile(
        schedule=schedule,
        activities=[torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=get_trace_fn(args),
    )


def benchmark_end_to_end(
    args: Namespace,
    pipeline_class: Type[Pipeline],
    text: List[str],
    generate_kwargs: dict,
) -> None:
    pipeline: Pipeline
    pipeline, initialization_time = run_and_log_time(partial(pipeline_class, args=args))

    warmup = args.warmup
    if warmup is None:
        warmup = args.profile

    all_metrics = []

    with (get_profiler(args) if args.profile else contextlib.nullcontext()) as p:
        for step in range(args.skip + warmup + args.cycles):
            generated_text, metrics = pipeline(text, **generate_kwargs)
            if args.profile:
                p.step()

            if step == 0:
                for i, o, _ in zip(text, generated_text, range(args.max_log_outputs)):
                    log_rank_n(f"{'-' * 60}\nINPUT = {i}\nOUTPUT = {o}", logger.info)

            if step >= args.skip + warmup:
                all_metrics.append(metrics)

            if args.clear_every_run:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()

    if len(all_metrics) > 0:
        log_rank_n("*** Performance metrics:", logger.info)
        log_dict(aggregate_and_format_metrics(all_metrics), logger.info)

    log_rank_n("*** Benchmarking stats:", logger.info)
    log_dict(
        {
            "Model initialization time": format_ms(initialization_time),
            "Model parameters": pipeline.get_num_parameters(),
            "Batch size": args.batch_size,
            **generate_kwargs,
        },
        logger.info,
    )


def aggregate_and_format_metrics(metrics: List[Dict[str, Any]]):
    all_metrics = {key: [metrics_[key] for metrics_ in metrics if key in metrics_] for key in METRIC_KEYS}
    mean_metrics = {key: np.mean(all_metrics[key]).item() for key in METRIC_KEYS if len(all_metrics[key]) > 0}
    throughput = mean_metrics[NUM_GENERATED_TOKENS] / mean_metrics[END_TO_END_TIME]
    model_throughput = mean_metrics[NUM_GENERATED_TOKENS] / mean_metrics[MODEL_TIME]

    return {
        "Latency (end to end)": format_ms(mean_metrics[END_TO_END_TIME]),
        "Latency (tokenization)": format_ms(mean_metrics[TOKENIZE_TIME]),
        "Latency (model)": format_ms(mean_metrics[MODEL_TIME]),
        "Latency (decode)": format_ms(mean_metrics[DECODE_TIME]),
        "Latency (max)": format_ms(max(all_metrics[END_TO_END_TIME])),
        "Latency (min)": format_ms(min(all_metrics[END_TO_END_TIME])),
        "Tokens generated": f"{mean_metrics[NUM_GENERATED_TOKENS]:.0f}",
        "Throughput (model)": f"{model_throughput:.2f} tokens/s",
        "Throughput (end to end)": f"{throughput:.2f} tokens/s",
        "Token time (end to end)": f"{format_ms(throughput ** -1)}/token",
    }
