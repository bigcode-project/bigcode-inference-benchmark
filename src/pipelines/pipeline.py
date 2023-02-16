import contextlib
import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.fast_init import fast_init
from src.utils.logging import format_ms, log_rank_n
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)


logger = logging.getLogger(__name__)

NUM_GENERATED_TOKENS = "num_generated_tokens"
TOKENIZE_TIME = "tokenize_time"
MODEL_TIME = "model_time"
DECODE_TIME = "decode_time"
END_TO_END_TIME = "end_to_end_time"

METRIC_KEYS = (
    NUM_GENERATED_TOKENS,
    TOKENIZE_TIME,
    MODEL_TIME,
    DECODE_TIME,
    END_TO_END_TIME,
)


class Pipeline:
    def __init__(
        self,
        *,
        model_type: Optional[str] = None,
        pretrained_model: Optional[str] = None,
        config_args: Dict[str, Any],
        tokenizer: str,
        device: torch.device,
        dtype: torch.dtype,
        fast_init: bool = True,
    ):
        self.initialization_metrics = {}
        log_rank_n("*** Setting up tokenizer", logger.info)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        t1 = time.perf_counter()

        self.device = device
        self.dtype = dtype
        self.is_int8 = self.dtype == torch.int8
        self.fast_init = fast_init
        if self.is_int8 and self.device != torch.device("cuda"):
            raise ValueError(f"Model quantization not supported on device {self.device}")

        self.config = self._get_config(model_type, pretrained_model, config_args)
        t2 = time.perf_counter()

        logger.info(f"Model configuration: {self.config}")

        if pretrained_model is None:
            self.model = self._create_model()
            if self.is_int8:
                self._reload_model()
        else:
            self.model = self._load_pretrained(pretrained_model)

        self.model.eval()
        t3 = time.perf_counter()
        self.initialization_metrics["tokenizer"] = t1 - t0
        self.initialization_metrics["configuration"] = t2 - t1
        self.initialization_metrics["total"] = t3 - t0

    def _create_model(self) -> PreTrainedModel:
        t0 = time.perf_counter()
        log_rank_n("*** Creating model", logger.info)
        with fast_init(self.device) if self.fast_init else contextlib.nullcontext():
            torch_dtype = torch.float16 if self.is_int8 else self.dtype
            model = AutoModelForCausalLM.from_config(config=self.config, torch_dtype=torch_dtype)
        t1 = time.perf_counter()
        log_rank_n("*** Moving to device", logger.info)
        model.to(self.device)
        t2 = time.perf_counter()
        log_rank_n("*** Initializing weights", logger.info)
        # Initialization is ~1000x faster on GPU.
        model.init_weights()
        t3 = time.perf_counter()
        self.initialization_metrics["model initialization"] = t1 - t0
        self.initialization_metrics["move to device"] = t2 - t1
        self.initialization_metrics["initialize weights"] = t3 - t2
        return model

    def _reload_model(self):
        self._save_pretrained("tmp")
        del self.model
        gc.collect()
        self.model = self._load_pretrained("tmp")

    def _save_pretrained(self, pretrained_model: str):
        t0 = time.perf_counter()
        log_rank_n(f"*** Saving model to {pretrained_model}", logger.info)
        t1 = time.perf_counter()
        self.initialization_metrics["save model"] = t1 - t0
        self.model.save_pretrained(pretrained_model)

    def _load_pretrained(self, pretrained_model: str) -> PreTrainedModel:
        t0 = time.perf_counter()
        log_rank_n(f"*** Loading model from {pretrained_model}", logger.info)
        kwargs = {"load_in_8bit": True, "device_map": "auto"} if self.is_int8 else {"torch_dtype": self.dtype}
        with fast_init(self.device) if self.fast_init else contextlib.nullcontext():
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                config=self.config,
                **kwargs,
            )
        t1 = time.perf_counter()
        self.initialization_metrics["load pretrained model"] = t1 - t0
        if not self.is_int8:
            log_rank_n("*** Moving to device", logger.info)
            model = model.to(self.device)
            t2 = time.perf_counter()
            self.initialization_metrics["move to device"] = t2 - t1
        return model

    def _get_config(
        self,
        model_type: Optional[str],
        pretrained_model: Optional[str],
        config_args: Dict[str, Any],
    ) -> PretrainedConfig:
        config_args = {
            "use_cache": True,
            "return_unused_kwargs": True,
            **config_args,
        }

        if model_type is None:
            if pretrained_model is None:
                raise ValueError("You need to provide either --model_type or --pretrained_model")
            config_class = AutoConfig
        elif model_type not in CONFIG_MAPPING:
            raise ValueError(f"Unknown model type: {model_type}")
        else:
            config_class = CONFIG_MAPPING[model_type]

        if pretrained_model is None:
            config_args.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "vocab_size": len(self.tokenizer),
                }
            )
            config, unused = config_class.from_dict({}, **config_args)
        else:
            config, unused = config_class.from_pretrained(pretrained_model, **config_args)

        if unused:
            raise ValueError(f"There were unused configuration parameters: {tuple(unused)}")

        return config

    def __call__(self, text: List[str], **generate_kwargs) -> Tuple[List[str], Dict[str, Any]]:
        t0 = time.perf_counter()
        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        t1 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**input_tokens, return_dict_in_generate=True, **generate_kwargs)
        t2 = time.perf_counter()

        output_tokens = output.sequences

        num_generated_tokens = sum(o.shape[0] - i.shape[0] for i, o in zip(input_tokens.input_ids, output_tokens))

        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        t3 = time.perf_counter()

        metrics = {
            NUM_GENERATED_TOKENS: num_generated_tokens,
            TOKENIZE_TIME: t1 - t0,
            MODEL_TIME: t2 - t1,
            DECODE_TIME: t3 - t2,
            END_TO_END_TIME: t3 - t0,
        }

        return output_text, metrics

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def aggregate_and_format_metrics(self, metrics: List[Dict[str, Any]]):
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
            "Tokens generated (average)": f"{mean_metrics[NUM_GENERATED_TOKENS]:.0f}",
            "Tokens generated (total)": f"{np.sum(all_metrics[NUM_GENERATED_TOKENS]).item():.0f}",
            "Throughput (model)": f"{model_throughput:.2f} tokens/s",
            "Throughput (end to end)": f"{throughput:.2f} tokens/s",
            "Token time (end to end)": f"{format_ms(throughput ** -1)}/token",
        }

    def get_initialization_metrics(self):
        return {f"Initialization time ({key})": format_ms(value) for key, value in self.initialization_metrics.items()}
