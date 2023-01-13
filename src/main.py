import src.pipelines
from src.utils import benchmark_end_to_end, get_arg_parser, get_dummy_batch
from src.utils.utils import configure_logging


def main() -> None:
    args = get_arg_parser().parse_args()

    inputs = get_dummy_batch(args.batch_size, args.max_input_length)

    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    pipeline_class = getattr(src.pipelines, args.pipeline_class)
    benchmark_end_to_end(args, pipeline_class, inputs, generate_kwargs)


if __name__ == "__main__":
    configure_logging()
    main()
