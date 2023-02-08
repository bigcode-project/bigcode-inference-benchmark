import os
from argparse import Namespace

import deepspeed
import torch

from src.pipelines.pipeline import Pipeline
from src.utils.arguments import check_unused


class DS_Pipeline(Pipeline):
    def __init__(self, args: Namespace) -> None:
        if args.model_class.lower() not in ["bloom", "gpt2"]:
            raise NotImplementedError(f"Deepspeed does not support the model {args.model_class}")

        check_unused(args, {"device": torch.device("cuda")}, enforce=True)
        # TODO: Works with other dtypes?
        check_unused(args, {"dtype": torch.float16})
        super().__init__(args)

        self.model = deepspeed.init_inference(
            self.model,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            # base_dir="./",
            dtype=args.dtype,
            replace_with_kernel_inject=args.inject_kernel,
            enable_cuda_graph=args.cuda_graph,
        )
