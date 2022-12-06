import pipelines
from utils import benchmark_end_to_end, get_arg_parser, get_args, get_dummy_batch, get_dummy_batch_tokenizer
from transformers import AutoTokenizer

def main() -> None:
    # deepspeed.init_distributed("nccl")

    args = get_args(get_arg_parser())

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    inputs = get_dummy_batch_tokenizer(args.batch_size, tokenizer, args.max_input_length)

    generate_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

    pipeline_class = getattr(pipelines, args.pipeline_class)
    benchmark_end_to_end(args, pipeline_class, inputs, generate_kwargs)


if __name__ == "__main__":
    main()
