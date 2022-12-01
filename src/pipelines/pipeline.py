from argparse import Namespace
from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer, BloomConfig, GPT2Config


class Pipeline:
    def __init__(self, args: Namespace) -> None:
        self.config, self.tokenizer = get_config_tokenizer(args)
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


def get_config_tokenizer(args: Namespace) -> Union[BloomConfig, GPT2Config]:
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")

    if args.model_class.lower() == "bloom":
        config = BloomConfig(
            attention_softmax_in_fp32=True,
            hidden_size=args.hidden_size,
            n_head=args.n_head,
            n_layer=args.n_layer,
            vocab_size=len(tokenizer),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elif args.model_class.lower() == "gpt2":
        config = GPT2Config(
            n_embd=args.hidden_size,
            n_head=args.n_head,
            n_layer=args.n_layer,
            n_positions=args.n_positions,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attention_type=args.attention_type,
            print_details=False,
            vocab_size=len(tokenizer),
        )

    return config, tokenizer
