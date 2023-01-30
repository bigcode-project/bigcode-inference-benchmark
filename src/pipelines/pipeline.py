import gc
import logging
import time
from argparse import Namespace
from typing import Any, Dict, List, Tuple, Type

import torch

from src.constants import DECODE_TIME, END_TO_END_TIME, MODEL_TIME, NUM_GENERATED_TOKENS, TOKENIZE_TIME
from src.utils.arguments import check_unused
from src.utils.fast_init import fast_init
from src.utils.logging import log_rank_n
from transformers import AutoTokenizer, BloomForCausalLM, GPT2LMHeadModel, PretrainedConfig, PreTrainedModel


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, args: Namespace) -> None:
        log_rank_n("*** Setting up tokenizer", logger.info)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.device = args.device

        model_class, config = self.get_config(args)
        is_int8 = args.dtype == torch.int8
        if is_int8:
            check_unused(args, {"device": torch.device("cuda")}, enforce=True)
        torch_dtype = torch.float16 if is_int8 else args.dtype

        log_rank_n("*** Creating model", logger.info)
        with fast_init(self.device):
            self.model = model_class._from_config(config=config, torch_dtype=torch_dtype)
        log_rank_n("*** Moving to device", logger.info)
        self.model.to(self.device)
        log_rank_n("*** Initializing weights", logger.info)
        # Initialization is ~1000x faster on GPU.
        self.model.init_weights()

        # Int8 can only be obtained by reloading a pretrained model
        if is_int8:
            log_rank_n("*** Saving model", logger.info)
            self.model.save_pretrained("tmp")
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            log_rank_n("*** Reloading model in int8", logger.info)
            with fast_init(self.device):
                self.model = model_class.from_pretrained(
                    "tmp",
                    load_in_8bit=True,
                    device_map="auto",
                )

        self.model.eval()

    def get_config(self, args) -> Tuple[Type[PreTrainedModel], PretrainedConfig]:
        config_args = {
            "activation_function": args.activation_function,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "vocab_size": len(self.tokenizer),
            "use_cache": True,
        }
        if args.model_class.lower() == "bloom":
            check_unused(args, {"attention_type": 1, "n_positions": None})
            config_args["attention_softmax_in_fp32"] = True
            config_args["hidden_size"] = args.hidden_size
            model_class = BloomForCausalLM
        elif args.model_class.lower() == "gpt2":
            config_args["attention_type"] = args.attention_type
            config_args["n_embd"] = args.hidden_size
            config_args["n_positions"] = args.n_positions
            config_args["pre_allocate_cache"] = args.pre_allocate_cache
            model_class = GPT2LMHeadModel
        else:
            raise NotImplementedError()

        return model_class, model_class.config_class(**config_args)

    def __call__(self, text: List[str], **generate_kwargs) -> Tuple[List[str], Dict[str, Any]]:
        t0 = time.perf_counter()
        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)

        t1 = time.perf_counter()
        with torch.no_grad():
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
