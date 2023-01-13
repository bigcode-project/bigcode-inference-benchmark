from src.pipelines import get_pipeline_class
from src.utils.arguments import get_arg_parser
from src.utils.benchmark import benchmark_end_to_end
from src.utils.input import get_dummy_batch
from src.utils.utils import configure_logging


def main() -> None:
    args = get_arg_parser().parse_args()

    inputs = get_dummy_batch(args.batch_size, args.max_input_length)

    generate_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}

    pipeline_class = get_pipeline_class(args.pipeline_class)
    benchmark_end_to_end(args, pipeline_class, inputs, generate_kwargs)


if __name__ == "__main__":
    configure_logging()
    main()
