from typing import List, Optional

from src.pipelines import get_pipeline_class
from src.utils.arguments import parse_args
from src.utils.benchmark import benchmark_end_to_end
from src.utils.input import get_dummy_batch
from src.utils.logging import configure_logging


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv=argv)

    inputs = get_dummy_batch(args.batch_size, args.max_input_length)

    generate_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    pipeline_class = get_pipeline_class(args.pipeline_class)
    benchmark_end_to_end(args, pipeline_class, inputs, generate_kwargs)


if __name__ == "__main__":
    configure_logging()
    main()
