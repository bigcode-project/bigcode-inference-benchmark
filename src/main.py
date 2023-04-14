import contextlib
import gc
import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

import torch

from src.metrics import Metrics
from src.pipeline import Pipeline, get_pipeline_class
from src.profile import get_profiler, logger
from src.utils import configure_logging, get_dummy_batch, log_dict, log_rank_n, parse_config_args


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
    parser.add_argument("-c", "--custom_generate", action="store_true")
    parser.add_argument("--pipeline_class", default="HF_Pipeline")
    parser.add_argument("--device", default="cuda", type=torch.device)
    parser.add_argument("--dtype", default="float16", type=lambda x: getattr(torch, x))
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--no_fast_init","--nf", dest="fast_init", action="store_false")
    parser.add_argument("--no_cache","--nc", dest="use_cache", action="store_false")
    parser.add_argument("--no_prefill","--np", dest="do_prefill", action="store_false")

    # Input and output
    parser.add_argument("--batch_size","-b", default=1, type=int)
    parser.add_argument("--max_input_length","-i", default=-1, type=int)
    parser.add_argument("--max_new_tokens","-g", default=100, type=int)

    # Cleanup
    parser.add_argument("--clear_every_run", action="store_true")

    # Benchmark cycles
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=5)

    # Profiling and logging
    parser.add_argument("--max_log_outputs", type=int)
    parser.add_argument("--breakdown_latency","--bl", action="store_true")
    parser.add_argument("--profile","-p", action="store_true")
    parser.add_argument("--profile_cycles","--pc", type=int)
    parser.add_argument("--full_trace","--pt", action="store_true")
    parser.add_argument("--show_op_names","--pn", action="store_true")
    parser.add_argument("--save", type=Path)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    t0 = time.perf_counter()
    parser = get_arg_parser()
    args = parser.parse_args(argv)
    config_args = parse_config_args(args.config_args)
    inputs = get_dummy_batch(args.batch_size, args.max_input_length)
    separate_profile = args.profile and args.profile_cycles is not None
    warmup = args.profile if args.warmup is None else args.warmup
    if separate_profile:
        pre_warmup_cycles = args.cycles
        post_warmup_cycles = args.profile_cycles
        benchmark_begin = args.skip
    else:
        pre_warmup_cycles = 0
        post_warmup_cycles = args.cycles
        benchmark_begin = args.skip + warmup
    benchmark_end = benchmark_begin + args.cycles

    max_log_outputs = args.batch_size if args.max_log_outputs is None else args.max_log_outputs

    pipeline_class = get_pipeline_class(args.pipeline_class)
    pipeline: Pipeline = pipeline_class(
        model_type=args.model_type,
        pretrained_model=args.pretrained_model,
        pretrained_config=args.pretrained_config,
        config_args=config_args,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        fast_init=args.fast_init,
        trust_remote_code=args.trust_remote_code,
        custom_generate=args.custom_generate,
        use_cache=args.use_cache,
        do_prefill=args.do_prefill,
        breakdown_latency=args.breakdown_latency,
    )

    all_metrics = []

    if args.profile:
        profiler = get_profiler(
            skip=args.skip + pre_warmup_cycles,
            warmup=warmup,
            cycles=post_warmup_cycles,
            full_trace=args.full_trace,
            show_op_names=args.show_op_names,
        )
    else:
        profiler = contextlib.nullcontext()

    benchmark_metrics = {
        "max_new_tokens": args.max_new_tokens,
        "Model parameters": pipeline.get_num_parameters(),
        "Cycles (warmup)": args.skip + warmup,
        "Cycles (benchmark)": args.cycles,
    }
    if args.profile:
        benchmark_metrics["Cycles (profile)"] = post_warmup_cycles
    benchmark_metrics["Cycles (total)"] = args.skip + warmup + pre_warmup_cycles + post_warmup_cycles

    if pipeline.device.type == "cuda":
        benchmark_metrics[Metrics.MEMORY_USED_INIT] = torch.cuda.memory_allocated()
        benchmark_metrics[Metrics.MEMORY_RESERVED_INIT] = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()

    t1 = time.perf_counter()
    with profiler as p:
        for step in range(args.skip + warmup + args.cycles):
            if step == args.skip + warmup:
                t2 = time.perf_counter()
                benchmark_metrics[Metrics.RUNTIME_WARMUP] = t2 - t1
            generated_text, metrics = pipeline(inputs, args.max_new_tokens)
            if args.profile:
                p.step()

            if step == 0:
                for i, o, _ in zip(inputs, generated_text, range(max_log_outputs)):
                    log_rank_n(f"{'-' * 60}\nINPUT = {i}\nOUTPUT = {o}", logger.info)

            if benchmark_begin <= step < benchmark_end:
                all_metrics.append(metrics)

            if args.clear_every_run:
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
    if pipeline.device.type == "cuda":
        benchmark_metrics[Metrics.MEMORY_USED_END] = torch.cuda.memory_allocated()
        benchmark_metrics[Metrics.MEMORY_RESERVED_END] = torch.cuda.memory_reserved()
        benchmark_metrics[Metrics.MEMORY_USED_MAX] = torch.cuda.max_memory_allocated()
        benchmark_metrics[Metrics.MEMORY_RESERVED_MAX] = torch.cuda.max_memory_reserved()

    t3 = time.perf_counter()
    benchmark_metrics[Metrics.RUNTIME_BENCHMARK] = t3 - t2
    benchmark_metrics[Metrics.RUNTIME_TOTAL] = t3 - t0

    if len(all_metrics) > 0:
        benchmark_metrics.update(pipeline.aggregate_metrics(all_metrics))

    benchmark_metrics = Metrics.reorder_metrics(benchmark_metrics)

    log_rank_n("*** Benchmark results:", logger.info)
    log_dict(Metrics.format_metrics(benchmark_metrics), logger.info)

    if args.save:
        save_path = Path(args.save).resolve()
        print(f"*** Saving results to {save_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as f:
            json.dump(
                {
                    "config": pipeline.config.to_dict(),
                    "results": benchmark_metrics,
                },
                f,
            )


if __name__ == "__main__":
    configure_logging()
    main()
