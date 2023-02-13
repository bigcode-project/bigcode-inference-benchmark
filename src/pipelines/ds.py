import os
from argparse import Namespace

import deepspeed
import torch

from src.pipelines.pipeline import Pipeline


class DS_Pipeline(Pipeline):
    def __init__(self, args: Namespace):
        if args.model_class.lower() not in ["bloom", "gpt2"]:
            raise NotImplementedError(f"Deepspeed does not support the model {args.model_class}")

        if args.device != torch.device("cuda"):
            raise ValueError(f"Deepspeed does not support device {args.device}")

        if args.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Deepspeed does not support dtype {args.dtype}")

        super().__init__(args)

        self.model = deepspeed.init_inference(
            self.model,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            # base_dir="./",
            dtype=args.dtype,
            replace_with_kernel_inject=args.inject_kernel,
        )

    def _get_config(self):
        config = super()._get_config()
        if config.model_type not in ("bloom", "gpt2"):
            raise ValueError(f"Deepspeed does not support model type {config.model_type}")
