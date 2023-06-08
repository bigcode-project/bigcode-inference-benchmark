
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
FILE_PREFIX=${7:-""}
CYCLES=${8:-10}

SAVE_DIR=data/benchmarks/v2
#BATCH_SIZES="1 2 4 8 16 24 32 48 64 96 128 160 224 256"
RUN="python3 src/main.py --max_log_outputs=0 --dtype=float16 --device=cuda  --custom_generate  --breakdown_latency --ignore_oom"


RUNTIME=("" "pre_allocate_kv_cache=True" "pre_allocate_kv_cache=True inference_runner=3")
RUNTIME_NAMES=("base" "pre_allocate" "graph")

ATTN=( \
  "attention_implementation=0" \
  "attention_implementation=1" \
  "attention_implementation=1 --pad_generated_tokens=0.5" \
  "attention_implementation=2" \
  "attention_implementation=0 fused_softmax=False" \
  "attention_implementation=0 fused_softmax=True" \
  "attention_implementation=3" \
  "attention_implementation=4" \
  "attention_implementation=5" \
  )
ATTN_NAME=( \
  "default" \
  "flash" \
  "flash_unpad_50" \
  "torch" \
  "no_jit" \
  "jit" \
  "torchflash" \
  "torchmem" \
  "torchcpp" \
  )


STEP=("--no_prefill" "--no_cache")
STEP_NAME=("decode" "prefill")

COMMON="--pretrained_model=$MODEL_PATH --tokenizer=$MODEL_PATH --cycles=$CYCLES --max_input_length=1 --max_new_tokens=$MAX_NEW_TOKENS --key_length_step=$TOKEN_STEP --batch_size=$BATCH_SIZE predict_last_token=True"

run () { # run(step, runtime, attn)
  FILE_NAME="$SAVE_DIR"/"$MODEL_NAME"_bs_"$BATCH_SIZE"_tok_"$MAX_NEW_TOKENS"_step_"$TOKEN_STEP"_"${STEP_NAME[$1]}"/"$FILE_PREFIX""${RUNTIME_NAMES[$2]}"_"${ATTN_NAME[$3]}".json
  if [ -f "$FILE_NAME" ];
  then
    echo "Skipping existing $FILE_NAME"
  else
    CMD="$RUN $COMMON  ${RUNTIME[$2]} ${ATTN[$3]} ${STEP[$1]} --save=$FILE_NAME"
    echo "$CMD"
    $CMD
  fi
}

if [ "${STEP_ID}" -eq "0" ]
then
  # Decode (default attn only)
  for runtime in {0..2}
  do
    run 0 $runtime 0
  done
else
  # Prefill (all runtimes are the same)
  for attn in {0..2}
  do
    run 1 0 $attn
  done
fi
