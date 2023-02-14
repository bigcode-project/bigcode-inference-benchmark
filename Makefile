check_dirs := src scripts

.PHONY: style
style:
	black --preview $(check_dirs)
	isort $(check_dirs)

BATCH_SIZE ?= 1
DTYPE ?= float16
HIDDEN_SIZE ?= 2048
N_HEAD ?= 16
N_LAYER ?= 24
N_POSITION ?= 2048
MAX_INPUT_LENGTH ?= -1

RUN_HF := python3 src/main.py --pipeline_class=HF_Pipeline
RUN_DF := deepspeed --num_gpus 1 src/main.py --pipeline_class=DF_Pipeline
EXP_ARGS := --dtype=${DTYPE} --batch_size=${BATCH_SIZE} --max_input_length=${MAX_INPUT_LENGTH}
COMMON_ARGS := n_head=${N_HEAD} n_layer=${N_LAYER} ${EXP_ARGS}
BLOOM_ARGS := --model_type=bloom hidden_size=${HIDDEN_SIZE} ${COMMON_ARGS}
GPT2_ARGS := --model_type=gpt2 n_embed=${HIDDEN_SIZE} ${COMMON_ARGS}
BIGCODE_ARGS := --model_type=gpt_bigcode n_embed=${HIDDEN_SIZE} ${COMMON_ARGS}


.PHONY: install
install:
	git submodule update --init
	pip install -r requirements.txt

.PHONY: bloom
bloom:
	${RUN_HF} ${BLOOM_ARGS}

.PHONY: bloom-ds
bloom-ds:
	${RUN_DS} ${BLOOM_ARGS}

.PHONY: gpt2-ds
gpt2:
	${RUN_HF} ${GPT2_ARGS}

.PHONY: gpt2-ds
gpt2-ds:
	${RUN_DS} ${GPT2_ARGS}

.PHONY: bigcode_mha
bigcode_mha:
	${RUN_HF} ${BIGCODE_ARGS} attention_type=1

.PHONY: bigcode_mqa1
bigcode_mqa1:
	${RUN_HF} ${BIGCODE_ARGS} attention_type=2

.PHONY: bigcode_mqa2
bigcode_mqa2:
	${RUN_HF} ${BIGCODE_ARGS} attention_type=3
