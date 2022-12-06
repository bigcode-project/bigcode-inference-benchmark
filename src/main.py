import pipelines
from utils import benchmark_end_to_end, get_arg_parser, get_args, get_dummy_batch


def main() -> None:
    # deepspeed.init_distributed("nccl")

    args = get_args(get_arg_parser())

    inputs = get_dummy_batch(args.batch_size, args.max_input_length)

    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    pipeline_class = getattr(pipelines, args.pipeline_class)
    benchmark_end_to_end(args, pipeline_class, inputs, generate_kwargs)


if __name__ == "__main__":
    main()
