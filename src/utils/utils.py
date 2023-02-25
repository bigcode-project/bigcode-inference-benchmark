import time
from functools import partial
from typing import Any, List, Optional, Tuple, Union


def run_and_log_time(execs: Union[List[partial], partial]) -> Tuple[Union[List[Any], Any], float]:
    # runs a function / list of functions and times them
    start_time = time.perf_counter()

    if type(execs) == list:
        results = []
        for f in execs:
            results.append(f())
    else:
        results = execs()

    time_elapsed = time.perf_counter() - start_time
    return results, time_elapsed


def parse_revision(pretrained_model: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    revision = None
    if pretrained_model is not None:
        pretrained_split = pretrained_model.split(":", 1)
        if len(pretrained_split) == 2:
            pretrained_model, revision = pretrained_split
    return pretrained_model, revision
