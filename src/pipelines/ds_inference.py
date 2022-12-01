import os
from argparse import Namespace

import deepspeed
import torch

from .pipeline import Pipeline


class DS_Inference_Pipeline(Pipeline):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        world_size = int(os.getenv("WORLD_SIZE", "1"))

        # with deepspeed.OnDevice(dtype=torch.bfloat16, device="meta"):
        #     model = BloomForCausalLM._from_config(config, torch_dtype=torch.bfloat16)
        self.model = self.model_class._from_config(self.config, torch_dtype=torch.bfloat16)
        self.model.eval()

        # checkpoints_json = os.path.join(args.model_name, "checkpoints.json")

        # if dist.get_rank() == 0:
        #     with io.open(checkpoints_json, "w", encoding="utf-8") as f:
        #         checkpoint_files = [str(entry) for entry in Path(args.model_name).rglob("*.[bp][it][n]") if entry.is_file()]
        #         data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
        #         json.dump(data, f)
        # dist.barrier()

        self.model = deepspeed.init_inference(
            self.model,
            mp_size=world_size,
            # base_dir="./",
            dtype=torch.float16,
            replace_with_kernel_inject=True
            # checkpoint=checkpoints_json,
        )

        self.input_device = torch.cuda.current_device()
