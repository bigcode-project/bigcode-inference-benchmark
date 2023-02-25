import contextlib
import gc
import time
from argparse import ArgumentParser, Namespace
from typing import List, Optional

import torch

from src.pipeline import get_pipeline_class
from src.profile import get_profiler, logger
from src.utils import (
    configure_logging,
    format_mib,
    format_ms,
    get_dummy_batch,
    log_dict,
    log_rank_n,
    parse_config_args,
)


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


def main(argv: Optional[List[str]] = None) -> None:
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    config_args = parse_config_args(args.config_args)
    generate_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    inputs = get_dummy_batch(args.batch_size, args.max_input_length)
    warmup = args.profile if args.warmup is None else args.warmup
    max_log_outputs = args.batch_size if args.max_log_outputs is None else args.max_log_outputs

    pipeline_class = get_pipeline_class(args.pipeline_class)
    pipeline = pipeline_class(
        model_type=args.model_type,
        pretrained_model=args.pretrained_model,
        pretrained_config=args.pretrained_config,
        config_args=config_args,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        fast_init=args.fast_init,
        trust_remote_code=args.trust_remote_code,
    )

    all_metrics = []

    if args.profile:
        profiler = get_profiler(
            skip=args.skip,
            warmup=warmup,
            cycles=args.cycles,
            full_trace=args.full_trace,
            show_op_names=args.show_op_names,
        )
    else:
        profiler = contextlib.nullcontext()

    benchmark_stats = {
        "Model parameters": pipeline.get_num_parameters(),
        "Batch size": len(inputs),
        **generate_kwargs,
        **pipeline.get_initialization_metrics(),
        "Warmup cycles": args.skip + warmup,
        "Benchmark cycles": args.cycles,
        "Total cycles": args.skip + warmup + args.cycles,
    }

    if pipeline.device.type == "cuda":
        benchmark_stats["Initial memory used"] = format_mib(torch.cuda.memory_allocated())
        benchmark_stats["Initial memory reserved"] = format_mib(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with profiler as p:
        for step in range(args.skip + warmup + args.cycles):
            if step == args.skip + warmup:
                t1 = time.perf_counter()
                benchmark_stats["Warmup time"] = format_ms(t1 - t0)
            generated_text, metrics = pipeline(inputs, **generate_kwargs)
            if args.profile:
                p.step()

            if step == 0:
                for i, o, _ in zip(inputs, generated_text, range(max_log_outputs)):
                    log_rank_n(f"{'-' * 60}\nINPUT = {i}\nOUTPUT = {o}", logger.info)

            if step >= args.skip + warmup:
                all_metrics.append(metrics)

            if args.clear_every_run:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
    if pipeline.device.type == "cuda":
        benchmark_stats["Memory used"] = format_mib(torch.cuda.memory_allocated())
        benchmark_stats["Memory reserved"] = format_mib(torch.cuda.memory_reserved())
        benchmark_stats["Max memory used"] = format_mib(torch.cuda.max_memory_allocated())
        benchmark_stats["Max memory reserved"] = format_mib(torch.cuda.max_memory_reserved())

    t2 = time.perf_counter()
    benchmark_stats["Benchmark time"] = format_ms(t2 - t1)
    benchmark_stats["Total time"] = format_ms(t2 - t0)

    if len(all_metrics) > 0:
        benchmark_stats.update(pipeline.aggregate_and_format_metrics(all_metrics))

    log_rank_n("*** Benchmark results:", logger.info)
    log_dict(benchmark_stats, logger.info)


if __name__ == "__main__":
    configure_logging()
    main()
