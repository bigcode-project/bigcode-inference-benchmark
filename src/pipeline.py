import contextlib
import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from optimum.onnxruntime import ORTModelForCausalLM
from src.fast_init import fast_init
from src.metrics import Metrics
from src.utils import log_rank_n, parse_revision
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        *,
        model_type: Optional[str] = None,
        pretrained_config: Optional[str] = None,
        pretrained_model: Optional[str] = None,
        config_args: Dict[str, Any],
        tokenizer: str,
        device: torch.device,
        dtype: torch.dtype,
        fast_init: bool = True,
        trust_remote_code: bool = False,
    ):
        self.global_metrics = {}
        log_rank_n("*** Setting up tokenizer", logger.info)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        t1 = time.perf_counter()

        self.device = device
        self.dtype = dtype
        self.is_int8 = self.dtype == torch.int8
        self.fast_init = fast_init
        self.trust_remote_code = trust_remote_code
        if self.is_int8 and self.device != torch.device("cuda"):
            raise ValueError(f"Model quantization not supported on device {self.device}")

        self.config = self._get_config(model_type, pretrained_config or pretrained_model, config_args)
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
        self.global_metrics[Metrics.INIT_TOKEN] = t1 - t0
        self.global_metrics[Metrics.INIT_CONFIG] = t2 - t1
        self.global_metrics[Metrics.INIT_TOTAL] = t3 - t0

    def _create_model(self) -> PreTrainedModel:
        t0 = time.perf_counter()
        log_rank_n("*** Creating model", logger.info)
        with fast_init(self.device) if self.fast_init else contextlib.nullcontext():
            torch_dtype = torch.float16 if self.is_int8 else self.dtype
            model = AutoModelForCausalLM.from_config(
                config=self.config, torch_dtype=torch_dtype, trust_remote_code=self.trust_remote_code
            )
        t1 = time.perf_counter()
        log_rank_n("*** Moving to device", logger.info)
        model.to(self.device)
        t2 = time.perf_counter()
        log_rank_n("*** Initializing weights", logger.info)
        # Initialization is ~1000x faster on GPU.
        model.init_weights()
        t3 = time.perf_counter()
        self.global_metrics[Metrics.INIT_CREATE] = t1 - t0
        self.global_metrics[Metrics.INIT_DEVICE] = t2 - t1
        self.global_metrics[Metrics.INIT_WEIGHTS] = t3 - t2

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
        self.global_metrics[Metrics.INIT_SAVE] = t1 - t0
        self.model.save_pretrained(pretrained_model)

    def _load_pretrained(self, pretrained_model: str) -> PreTrainedModel:
        t0 = time.perf_counter()
        log_rank_n(f"*** Loading model from {pretrained_model}", logger.info)
        kwargs = {"load_in_8bit": True, "device_map": "auto"} if self.is_int8 else {"torch_dtype": self.dtype}
        with fast_init(self.device) if self.fast_init else contextlib.nullcontext():
            pretrained_model, revision = parse_revision(pretrained_model)
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                revision=revision,
                config=self.config,
                trust_remote_code=self.trust_remote_code,
                **kwargs,
            )
        t1 = time.perf_counter()
        self.global_metrics["load pretrained model"] = t1 - t0
        if not self.is_int8:
            log_rank_n("*** Moving to device", logger.info)
            model = model.to(self.device)
            t2 = time.perf_counter()
            self.global_metrics[Metrics.INIT_DEVICE] = t2 - t1
        return model

    def _get_config(
        self,
        model_type: Optional[str],
        pretrained_config: Optional[str],
        config_args: Dict[str, Any],
    ) -> PretrainedConfig:
        config_args = {
            "use_cache": True,
            "return_unused_kwargs": True,
            **config_args,
        }

        if model_type is None:
            if pretrained_config is None:
                raise ValueError("You need to provide either --model_type or --pretrained_model")
            config_class = AutoConfig
        elif model_type not in CONFIG_MAPPING:
            raise ValueError(f"Unknown model type: {model_type}")
        else:
            config_class = CONFIG_MAPPING[model_type]
            config_args["model_type"] = model_type

        if pretrained_config is None:
            config_args.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "vocab_size": len(self.tokenizer),
                }
            )
            config, unused = config_class.from_dict({}, **config_args)
        else:
            pretrained_config, revision = parse_revision(pretrained_config)
            config, unused = config_class.from_pretrained(
                pretrained_config, revision=revision, trust_remote_code=self.trust_remote_code, **config_args
            )

        if unused:
            raise ValueError(f"There were unused configuration parameters: {tuple(unused)}")

        return config

    def __call__(self, text: List[str], **generate_kwargs) -> Tuple[List[str], Dict[str, Any]]:
        t0 = time.perf_counter()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)

        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in inputs.items()}

        t1 = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, return_dict_in_generate=True, **generate_kwargs)
        t2 = time.perf_counter()

        output_tokens = output.sequences

        batch_size, input_length = inputs["input_ids"].shape
        output_length = output_tokens.size(1)

        output_text = self.tokenizer.batch_decode(output_tokens.cpu(), skip_special_tokens=True)
        t3 = time.perf_counter()

        metrics = {
            Metrics.BATCH_SIZE: batch_size,
            Metrics.INPUT_LENGTH: input_length,
            Metrics.OUTPUT_LENGTH: output_length,
            Metrics.TOKENS_SAMPLE: output_length - input_length,
            Metrics.TOKENS_BATCH: batch_size * (output_length - input_length),
            Metrics.LATENCY_TOKEN: t1 - t0,
            Metrics.LATENCY_MODEL: t2 - t1,
            Metrics.LATENCY_DECODE: t3 - t2,
            Metrics.LATENCY_E2E: t3 - t0,
        }

        return output_text, metrics

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def aggregate_metrics(self, metrics: List[Dict[str, Any]]):
        all_metrics = {
            key: [metrics_[key] for metrics_ in metrics if key in metrics_]
            for key in (
                Metrics.BATCH_SIZE,
                Metrics.INPUT_LENGTH,
                Metrics.OUTPUT_LENGTH,
                Metrics.TOKENS_SAMPLE,
                Metrics.TOKENS_BATCH,
                Metrics.LATENCY_TOKEN,
                Metrics.LATENCY_MODEL,
                Metrics.LATENCY_DECODE,
                Metrics.LATENCY_E2E,
            )
        }
        mean_metrics = {key: np.mean(value).item() for key, value in all_metrics.items() if len(value) > 0}
        throughput = mean_metrics[Metrics.TOKENS_BATCH] / mean_metrics[Metrics.LATENCY_E2E]
        model_throughput = mean_metrics[Metrics.TOKENS_BATCH] / mean_metrics[Metrics.LATENCY_MODEL]

        return {
            **self.global_metrics,
            **mean_metrics,
            Metrics.LATENCY_MAX: max(all_metrics[Metrics.LATENCY_E2E]),
            Metrics.LATENCY_MIN: min(all_metrics[Metrics.LATENCY_E2E]),
            Metrics.LATENCY_STD: np.std(all_metrics[Metrics.LATENCY_E2E]).item(),
            Metrics.THROUGHPUT_MODEL: model_throughput,
            Metrics.THROUGHPUT_E2E: throughput,
            Metrics.TOKEN_TIME: throughput**-1,
        }


class HF_Pipeline(Pipeline):
    pass


class DS_Pipeline(Pipeline):
    def __init__(self, **kwargs):
        import deepspeed

        super().__init__(**kwargs)

        if self.device != torch.device("cuda"):
            raise ValueError(f"Deepspeed does not support device {self.device}")

        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Deepspeed does not support dtype {self.dtype}")

        if self.config.model_type not in ("bloom", "gpt2"):
            raise ValueError(f"Deepspeed does not support model type {self.config.model_type}")

        self.model = deepspeed.init_inference(
            self.model,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            # base_dir="./",
            dtype=self.dtype,
            replace_with_kernel_inject=True,
        )


class ONNX_Pipeline(Pipeline):
    def __init__(
        self,
        *,
        model_type: Optional[str] = None,
        pretrained_config: Optional[str] = None,
        pretrained_model: Optional[str] = None,
        config_args: Dict[str, Any],
        tokenizer: str,
        device: torch.device,
        dtype: torch.dtype,
        fast_init: bool = False,
        trust_remote_code: bool = False,
    ):
        self.global_metrics = {}
        log_rank_n("*** Setting up tokenizer", logger.info)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        t1 = time.perf_counter()

        self.device = device

        if device == torch.device("cpu"):
            provider = "CPUExecutionProvider"
        elif device == torch.device("cuda"):
            provider = "CUDAExecutionProvider"

        self.dtype = dtype

        if self.dtype != torch.float32:
            raise NotImplementedError(f"currently dtype {self.dtype} is not supported")

        self.config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=trust_remote_code)
        t2 = time.perf_counter()

        logger.info(f"Model configuration: {self.config}")

        self.model = ORTModelForCausalLM.from_pretrained(
            pretrained_model, provider=provider, trust_remote_code=trust_remote_code
        )
        t3 = time.perf_counter()
        self.global_metrics[Metrics.INIT_TOKEN] = t1 - t0
        self.global_metrics[Metrics.INIT_CONFIG] = t2 - t1
        self.global_metrics[Metrics.INIT_TOTAL] = t3 - t0

    def get_num_parameters(self) -> int:
        return "Not implemented"


_PIPELINE_CLASS_MAP = {
    "HF_Pipeline": HF_Pipeline,
    "DS_Pipeline": DS_Pipeline,
    "ONNX_Pipeline": ONNX_Pipeline,
}


def get_pipeline_class(name):
    if name not in _PIPELINE_CLASS_MAP:
        raise NotImplementedError(f"Unsupported pipeline class: {name}")
    return _PIPELINE_CLASS_MAP[name]
