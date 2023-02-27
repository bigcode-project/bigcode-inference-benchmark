from typing import Any, Callable, Dict


def format_round(x: float) -> str:
    return str(round(x))


def format_throughput(x: float) -> str:
    return f"{x:.2f} tokens/s"


def format_inverse_throughput(x: float) -> str:
    return f"{format_ms(x ** -1)}/token"


def format_ms(t: float) -> str:
    return f"{1000 * t:.2f} ms"


def format_mib(m: float) -> str:
    return f"{m/2**20:.0f} MiB"


class Metrics:
    LATENCY_E2E = "Latency (end to end)"
    LATENCY_TOKEN = "Latency (tokenization)"
    LATENCY_MODEL = "Latency (model)"
    LATENCY_DECODE = "Latency (decode)"
    LATENCY_MAX = "Latency (max)"
    LATENCY_MIN = "Latency (min)"
    LATENCY_STD = "Latency (std)"
    BATCH_SIZE = "Batch size"
    INPUT_LENGTH = "Input sequence length"
    OUTPUT_LENGTH = "Output sequence length"
    TOKENS_SAMPLE = "Tokens generated (sample)"
    TOKENS_BATCH = "Tokens generated (batch)"
    THROUGHPUT_MODEL = "Throughput (model)"
    THROUGHPUT_E2E = "Throughput (end to end)"
    TOKEN_TIME = "Token time (end to end)"
    INIT_TOKEN = "Initialization time (tokenizer)"
    INIT_CONFIG = "Initialization time (configuration)"
    INIT_DEVICE = "Initialization time (move to device)"
    INIT_TOTAL = "Initialization time (total)"
    INIT_CREATE = "Initialization time (create model)"
    INIT_WEIGHTS = "Initialization time (init weights)"
    INIT_SAVE = "Initialization time (save model)"
    INIT_LOAD = "Initialization time (load model)"
    RUNTIME_WARMUP = "Runtime time (warmup)"
    RUNTIME_BENCHMARK = "Runtime time (benchmark)"
    RUNTIME_TOTAL = "Runtime time (total)"
    MEMORY_USED_INIT = "Memory used (init)"
    MEMORY_USED_END = "Memory used (end)"
    MEMORY_USED_MAX = "Memory used (max)"
    MEMORY_RESERVED_INIT = "Memory reserved (init)"
    MEMORY_RESERVED_END = "Memory reserved (end)"
    MEMORY_RESERVED_MAX = "Memory reserved (max)"

    _METRIC_ORDER_AND_FORMAT: Dict[str, Callable[[Any], str]] = {
        LATENCY_E2E: format_ms,
        LATENCY_TOKEN: format_ms,
        LATENCY_MODEL: format_ms,
        LATENCY_DECODE: format_ms,
        LATENCY_MAX: format_ms,
        LATENCY_MIN: format_ms,
        LATENCY_STD: format_ms,
        BATCH_SIZE: format_round,
        INPUT_LENGTH: format_round,
        OUTPUT_LENGTH: format_round,
        TOKENS_SAMPLE: format_round,
        TOKENS_BATCH: format_round,
        THROUGHPUT_MODEL: format_throughput,
        THROUGHPUT_E2E: format_throughput,
        TOKEN_TIME: format_inverse_throughput,
        INIT_TOKEN: format_ms,
        INIT_CONFIG: format_ms,
        INIT_DEVICE: format_ms,
        INIT_TOTAL: format_ms,
        INIT_CREATE: format_ms,
        INIT_WEIGHTS: format_ms,
        INIT_SAVE: format_ms,
        INIT_LOAD: format_ms,
        RUNTIME_WARMUP: format_ms,
        RUNTIME_BENCHMARK: format_ms,
        RUNTIME_TOTAL: format_ms,
        MEMORY_USED_INIT: format_mib,
        MEMORY_USED_END: format_mib,
        MEMORY_USED_MAX: format_mib,
        MEMORY_RESERVED_INIT: format_mib,
        MEMORY_RESERVED_END: format_mib,
        MEMORY_RESERVED_MAX: format_mib,
    }

    @classmethod
    def reorder_metrics(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics = metrics.copy()
        reordered_metrics = {}
        for name, format_fn in cls._METRIC_ORDER_AND_FORMAT.items():
            if name in metrics:
                reordered_metrics[name] = metrics.pop(name)
        reordered_metrics.update(metrics)
        return reordered_metrics

    @classmethod
    def format_metrics(cls, metrics: Dict[str, Any]) -> Dict[str, str]:
        return {key: cls._METRIC_ORDER_AND_FORMAT.get(key, str)(value) for key, value in metrics.items()}
