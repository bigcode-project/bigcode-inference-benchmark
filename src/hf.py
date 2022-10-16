from argparse import Namespace

import torch
from transformers import BloomForCausalLM

import utils
from model import Model
from utils import benchmark_end_to_end, get_dummy_batch


class HFAccelerateModel(Model):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        model_kwargs = {}
        if args.dtype == torch.int8:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = args.dtype

        self.input_device = "cuda:0"
        self.model = BloomForCausalLM._from_config(self.config, **model_kwargs).to(self.input_device)
        self.model.eval()


def main() -> None:
    args = utils.get_args(utils.get_arg_parser())

    inputs = get_dummy_batch(args.batch_size)
    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    benchmark_end_to_end(args, HFAccelerateModel, inputs, generate_kwargs)


if __name__ == "__main__":
    main()
