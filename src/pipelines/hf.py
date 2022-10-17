from argparse import Namespace

import torch
from transformers import BloomForCausalLM

from .pipeline import Pipeline


class HF_Pipeline(Pipeline):
    def __init__(self, args: Namespace, device: str = "cpu") -> None:
        super().__init__(args)

        model_kwargs = {}
        if args.dtype == torch.int8:
            model_kwargs["load_in_8bit"] = True
        else:
            model_kwargs["torch_dtype"] = args.dtype

        self.input_device = device
        self.model = BloomForCausalLM._from_config(self.config, **model_kwargs).to(self.input_device)
        self.model.eval()


class HF_CPU_Pipeline(HF_Pipeline):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args, "cpu")


class HF_GPU_Pipeline(HF_Pipeline):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args, "cuda:0")
