import contextlib
import logging
from typing import Union

import torch

from src.utils import log_rank_n


logger = logging.getLogger(__name__)


def get_trace_fn(full_trace: bool = False, show_op_names: bool = False, rank: int = -1, cpu: bool = False):
    def trace_fn(
        p: torch.profiler.profile,
    ):
        averages = p.key_averages()
        var_name = f"self_{'cpu' if cpu else 'cuda'}_time_total"
        if full_trace:
            # Show every GPU op.
            # Exclude CPU cuda ops to shorten the table.
            events = torch.autograd.profiler.EventList(
                [evt for evt in p.profiler.function_events if getattr(evt, var_name) > 0]
            )
            log_rank_n(events.table(row_limit=-1, max_src_column_width=1000), logger.info, rank)

        if show_op_names:
            # Show non-cropped names, in the same order as in the table.
            averages_sorted = torch.autograd.profiler.EventList(
                sorted(averages, key=lambda evt: getattr(evt, var_name), reverse=True)
            )
            for entry in averages_sorted:
                log_rank_n(entry.key, logger.info, rank)

        # Try to avoid name cropping, still hard-coded to max 55 characters
        log_rank_n(averages.table(sort_by=var_name, row_limit=-1, max_src_column_width=1000), logger.info, rank)

        # Store results for future use.
        p.bc_profile_result = p.profiler.function_events

    return trace_fn


def get_profiler(
    skip: int,
    warmup: int,
    cycles: int,
    full_trace: bool = False,
    show_op_names: bool = False,
    cpu=False,
) -> Union[torch.profiler.profile, contextlib.nullcontext]:
    schedule = torch.profiler.schedule(
        # Warmup is a must if measuring speed as it's when all the optimizations are performed
        # e.g. on 8x80 a100 the first pass of 100 tokens takes 23sec, and the next one is 4secs
        skip_first=skip,
        # Warmup for the profiler
        warmup=warmup,
        wait=0,
        active=cycles,
    )
    return torch.profiler.profile(
        schedule=schedule,
        activities=[torch.profiler.ProfilerActivity.CPU if cpu else torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=get_trace_fn(full_trace, show_op_names, cpu=cpu),
        with_modules=True,
    )
