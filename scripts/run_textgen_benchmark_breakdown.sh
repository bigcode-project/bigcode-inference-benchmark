
# Santacoder prefill.
# ./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 32 2040 5 0
# Santacoder decode (fewer data points because slower)
# ./scripts/run_benchmark_breakdown.sh santacoder bigcode/gpt_bigcode-santacoder 32 2040 11 1
MODEL_NAME=${1:-"santacoder"}
MODEL_PATH=${2:-"bigcode/gpt_bigcode-santacoder"}
BATCH_SIZE=${3:-32}
MAX_NEW_TOKENS=${4:-2040}
# Prime number to see key length padding effect.
TOKEN_STEP=${5:-5}
STEP_ID=${6:-""}
CYCLES=${7:-10}

SAVE_DIR=data/benchmarks/v5
RUN="python3 -m src.main --pipeline_class=TG_Pipeline --max_log_outputs=0 --dtype=float16 --device=cuda  --custom_generate  --breakdown_latency --ignore_oom --no_fast_init "


IMPL=("flash" "causal" "vector" "bigcode" "bigcode2" "bigcode3")


STEP=("" "--no_cache")
STEP_NAME=("decode" "prefill")

COMMON="--pretrained_model=$MODEL_PATH --tokenizer=$MODEL_PATH --cycles=$CYCLES --max_input_length=1 --max_new_tokens=$MAX_NEW_TOKENS --key_length_step=$TOKEN_STEP --batch_size=$BATCH_SIZE"

run () { # run(step, runtime, attn)
  FILE_NAME="$SAVE_DIR"/"$MODEL_NAME"_bs_"$BATCH_SIZE"_tok_"$MAX_NEW_TOKENS"_"${STEP_NAME[$1]}"_step_"$TOKEN_STEP"_"$CYCLES"/"${IMPL[$2]}".json
  if [ -f "$FILE_NAME" ];
  then
    echo "Skipping existing $FILE_NAME"
  else
    export MODEL_TYPE="${IMPL[$2]}"
    CMD="$RUN $COMMON ${STEP[$1]} --save=$FILE_NAME"
    echo "MODEL_TYPE=${IMPL[$2]} $CMD"
    $CMD
  fi
}

for impl in {0..5}
do
  if [ "${STEP_ID}" -eq "0" ]
  then
    # Decode (default attn only)
    run 0 $impl
  else
    # Prefill
    run 1 $impl
  fi
done
