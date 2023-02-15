import os

import deepspeed
import torch

from src.pipelines.pipeline import Pipeline


class DS_Pipeline(Pipeline):
    def __init__(self, **kwargs):
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
