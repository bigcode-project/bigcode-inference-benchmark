import contextlib
import gc
import logging
import time
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch

from src.utils.arguments import check_unused
from src.utils.fast_init import fast_init
from src.utils.logging import format_ms, log_rank_n
from transformers import (
    AutoTokenizer,
    BloomForCausalLM,
    GPT2LMHeadModel,
    GPTBigCodeLMHeadModel,
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
    def __init__(self, args: Namespace) -> None:
        self.args = args
        log_rank_n("*** Setting up tokenizer", logger.info)
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.device = args.device
        self.is_int8 = args.dtype == torch.int8
        if self.is_int8:
            check_unused(args, {"device": torch.device("cuda")}, enforce=True)

        self.model_class, self.config = self._get_config()

        pretrained_path = args.pretrained_model
        if pretrained_path is None:
            self.model = self._create_model()
            if self.is_int8:
                self._save_and_reload()
        else:
            self.model = self._load_pretrained(pretrained_path)

        self.model.eval()

    def _create_model(self):
        log_rank_n("*** Creating model", logger.info)
        with fast_init(self.device) if self.args.fast_init else contextlib.nullcontext():
            torch_dtype = torch.float16 if self.is_int8 else self.args.dtype
            model = self.model_class._from_config(config=self.config, torch_dtype=torch_dtype)

        log_rank_n("*** Moving to device", logger.info)
        model.to(self.device)
        log_rank_n("*** Initializing weights", logger.info)
        # Initialization is ~1000x faster on GPU.
        model.init_weights()
        return model

    def _save_and_reload(self):
        self._save_pretrained("tmp")
        del self.model
        gc.collect()
        self._load_pretrained("tmp")

    def _save_pretrained(self, pretrained_path):
        log_rank_n(f"*** Saving model to {pretrained_path}", logger.info)
        self.model.save_pretrained(pretrained_path)

    def _load_pretrained(self, pretrained_path):
        log_rank_n(f"*** Loading model from {pretrained_path}", logger.info)
        with fast_init(self.device) if self.args.fast_init else contextlib.nullcontext():
            return self.model_class.from_pretrained(
                pretrained_path,
                config=self.config,
                load_in_8bit=self.is_int8,
                device_map="auto",
            )

    def _get_config(self) -> Tuple[Type[PreTrainedModel], PretrainedConfig]:
        config_args = {
            "activation_function": self.args.activation_function,
            "n_head": self.args.n_head,
            "n_layer": self.args.n_layer,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "vocab_size": len(self.tokenizer),
            "use_cache": True,
            # These are not used by all models, but ok to set anyway as they will just be ignored.
            "attention_type": self.args.attention_type,
            "cuda_graph": self.args.cuda_graph,
            "inference_runner": self.args.inference_runner,
        }

        if self.args.model_class.lower() == "bloom":
            config_args["attention_softmax_in_fp32"] = True
            config_args["hidden_size"] = self.args.hidden_size
            model_class = BloomForCausalLM
        elif self.args.model_class.lower() == "gpt2":
            config_args["n_embd"] = self.args.hidden_size
            config_args["n_positions"] = self.args.n_positions
            model_class = GPT2LMHeadModel
        elif self.args.model_class.lower() == "gpt_bigcode":
            config_args["n_embd"] = self.args.hidden_size
            config_args["n_positions"] = self.args.n_positions
            model_class = GPTBigCodeLMHeadModel
        else:
            raise NotImplementedError()
        # Use defaults or pretrained config for missing arguments.
        config_args = {key: value for key, value in config_args.items() if value is not None}
        if self.args.pretrained_model is None:
            config = model_class.config_class(**config_args)
        else:
            config = model_class.config_class.from_pretrained(self.args.pretrained_model, **config_args)

        return model_class, config

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

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_token_lengths = [x.shape[0] for x in output_tokens]
        num_generated_tokens = [o - i for i, o in zip(input_token_lengths, output_token_lengths)]

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
            "Tokens generated": f"{mean_metrics[NUM_GENERATED_TOKENS]:.0f}",
            "Throughput (model)": f"{model_throughput:.2f} tokens/s",
            "Throughput (end to end)": f"{throughput:.2f} tokens/s",
            "Token time (end to end)": f"{format_ms(throughput ** -1)}/token",
        }
