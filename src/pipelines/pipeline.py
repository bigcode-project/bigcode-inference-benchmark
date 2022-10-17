from argparse import Namespace
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, BloomConfig


class Pipeline:
    def __init__(self, args: Namespace) -> None:
        self.config = BloomConfig.from_dict(
            {
                "apply_residual_connection_post_layernorm": False,
                "architectures": ["BloomModel"],
                "attention_dropout": 0.0,
                "attention_softmax_in_fp32": True,
                "bias_dropout_fusion": True,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "hidden_dropout": 0.0,
                "hidden_size": args.hidden_size,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-05,
                "masked_softmax_fusion": True,
                "model_type": "bloom",
                "n_head": args.n_head,
                "n_inner": None,
                "n_layer": args.n_layer,
                "offset_alibi": 100,
                "pad_token_id": 3,
                "pretraining_tp": 1,
                "skip_bias_add": True,
                "skip_bias_add_qkv": False,
                "slow_but_exact": False,
                "transformers_version": "4.22.2",
                "unk_token_id": 0,
                "use_cache": True,
                "vocab_size": 250880,
            }
        )

        # hardcoded for now to bigscience/bloom
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

        self.model = None
        self.input_device = None

    def __call__(self, text: List[str], **generate_kwargs) -> Tuple[List[str], List[int]]:
        input_tokens = self.tokenizer(text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.input_device)

        with torch.no_grad():
            output = self.model.generate(**input_tokens, return_dict_in_generate=True, **generate_kwargs)

        output_tokens = output.sequences

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_token_lengths = [x.shape[0] for x in output_tokens]
        num_generated_tokens = [o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return output_text, num_generated_tokens

    def get_num_parameters(self) -> int:
        param_count = 0
        for i in self.model.parameters():
            param_count += i.numel()
        return param_count
