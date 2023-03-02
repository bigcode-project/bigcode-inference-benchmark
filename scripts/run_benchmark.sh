SAVE_DIR=data/benchmarks/v1
BATCH_SIZES="1 2 4 8 16 24 32 48 64 96 128 160 224 256"
RUN="python3 src/main.py --tokenizer=bigcode/santacoder --max_log_outputs=1 --dtype=float16 --device=cuda"
RUN_DEEPSPEED="deepspeed --num_gpus 1 src/main.py --pipeline_class=DS_Pipeline --tokenizer=bigcode/santacoder --max_log_outputs=1 --dtype=float16 --device=cuda"

SANTACODER="--pretrained_model=bigcode/santacoder --trust_remote_code"
GPT_BIGCODE="--pretrained_model=bigcode/santacoder-fast-inference:linear"
PRE_ALLOCATE="--pretrained_model=bigcode/santacoder-fast-inference:linear pre_allocate_kv_cache=True"
INFERENCE_RUNNER="--pretrained_model=bigcode/santacoder-fast-inference:linear pre_allocate_kv_cache=True inference_runner=1"
CUDA_GRAPH="--pretrained_model=bigcode/santacoder-fast-inference:linear pre_allocate_kv_cache=True inference_runner=3"
MHA_GPT2="--model_type=gpt2 n_positions=2048 n_embd=2048 n_head=16 n_layer=24"
MHA_GPT_BIGCODE="--model_type=gpt_bigcode attention_type=1 n_positions=2048 n_embd=2048 n_head=16 n_layer=24"
MHA_PRE_ALLOCATE="--model_type=gpt_bigcode n_positions=2048 n_embd=2048 n_head=16 n_layer=24 pre_allocate_kv_cache=True max_sequence_length=1024"
MQA2_GPT_BIGCODE="--model_type=gpt_bigcode attention_type=3 n_positions=2048 n_embd=2048 n_head=16 n_layer=24"
MQA2_PRE_ALLOCATE="--model_type=gpt_bigcode attention_type=3 n_positions=2048 n_embd=2048 n_head=16 n_layer=24 pre_allocate_kv_cache=True"

SEQ_TOK=("-1 1" "-1 100" "501 1" "504 1" "-1 1000")

for seq_tok in "${SEQ_TOK[@]}"
do
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/santacoder $RUN --cycles=10 $SANTACODER
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/gpt_bigcode $RUN --cycles=10 $GPT_BIGCODE
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/pre_allocate $RUN --cycles=10 $PRE_ALLOCATE
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/inference_runner $RUN --cycles=10 $INFERENCE_RUNNER
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/cuda_graph $RUN --cycles=10 $CUDA_GRAPH
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/mha_deepspeed $RUN_DEEPSPEED --cycles=10 $MHA_GPT2
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/mha_gpt2 $RUN --cycles=10 $MHA_GPT2
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/mha_gpt_bigcode $RUN --cycles=10 $MHA_GPT_BIGCODE
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/mha_pre_allocate $RUN --cycles=10 $MHA_PRE_ALLOCATE
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/mqa2_gpt_bigcode $RUN --cycles=10 $MQA2_GPT_BIGCODE
  ./scripts/run_grid.sh "$BATCH_SIZES" $seq_tok $SAVE_DIR/mqa2_pre_allocate $RUN --cycles=10 $MQA2_PRE_ALLOCATE
done
