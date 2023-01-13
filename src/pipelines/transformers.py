from argparse import Namespace

from src.pipelines.pipeline import Pipeline, check_unused


class HF_Pipeline(Pipeline):
    def __init__(self, args: Namespace) -> None:
        check_unused(args, {"inject_kernel": False, "cuda_graph": False})
        super().__init__(args)
