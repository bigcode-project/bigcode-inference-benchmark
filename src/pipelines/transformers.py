from argparse import Namespace

from src.pipelines.pipeline import Pipeline


class HF_Pipeline(Pipeline):
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
