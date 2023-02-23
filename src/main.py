from typing import List, Optional

from src.pipelines import get_pipeline_class
from src.utils.arguments import parse_args
from src.utils.benchmark import benchmark_end_to_end
from src.utils.input import get_dummy_batch
from src.utils.logging import configure_logging


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv=argv)

    pipeline_class = get_pipeline_class(args.pipeline_class)
    pipeline = pipeline_class(
        model_type=args.model_type,
        pretrained_model=args.pretrained_model,
        pretrained_config=args.pretrained_config,
        config_args=args.config_args,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        fast_init=args.fast_init,
        trust_remote_code=args.trust_remote_code,
    )

    benchmark_end_to_end(
        pipeline=pipeline,
        inputs=get_dummy_batch(args.batch_size, args.max_input_length),
        generate_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        profile=args.profile,
        skip=args.skip,
        warmup=args.profile if args.warmup is None else args.warmup,
        cycles=args.cycles,
        full_trace=args.full_trace,
        show_op_names=args.show_op_names,
        max_log_outputs=args.batch_size if args.max_log_outputs is None else args.max_log_outputs,
        clear_every_run=args.clear_every_run,
    )


if __name__ == "__main__":
    configure_logging()
    main()
