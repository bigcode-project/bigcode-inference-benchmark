check_dirs := src scripts

.PHONY: style
style:
	black --preview $(check_dirs)
	isort $(check_dirs)

BATCH_SIZE ?= 1
DTYPE ?= float32
HIDDEN_SIZE ?= 2048
N_HEAD ?= 16
N_LAYER ?= 24
N_POSITION ?= 2048
MAX_INPUT_LENGTH ?= -1

RUN_HF := python3 src/main.py --pipeline_class=HF_Pipeline
RUN_DS := deepspeed --num_gpus 1 src/main.py --pipeline_class=DS_Pipeline
EXP_ARGS := --dtype=${DTYPE} --batch_size=${BATCH_SIZE} --max_input_length=${MAX_INPUT_LENGTH} ${EXTRA_ARGS}
COMMON_ARGS :=  ${EXP_ARGS} n_head=${N_HEAD} n_layer=${N_LAYER}
BLOOM_ARGS := --model_type=bloom ${COMMON_ARGS} hidden_size=${HIDDEN_SIZE}
GPT2_ARGS := --model_type=gpt2 ${COMMON_ARGS} n_embd=${HIDDEN_SIZE}
BIGCODE_ARGS := --model_type=gpt_bigcode ${COMMON_ARGS} n_embd=${HIDDEN_SIZE}


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

.PHONY: gpt2
gpt2:
	${RUN_HF} ${GPT2_ARGS}

.PHONY: gpt2-ds
gpt2-ds:
	${RUN_DS} ${GPT2_ARGS}

.PHONY: gpt-bigcode-mha
gpt-bigcode-mha:
	${RUN_HF} ${BIGCODE_ARGS} attention_type=1

.PHONY: gpt-bigcode-mqa1
gpt-bigcode-mqa1:
	${RUN_HF} ${BIGCODE_ARGS} attention_type=2

.PHONY: gpt-bigcode-mqa2
gpt-bigcode-mqa2:
	${RUN_HF} ${BIGCODE_ARGS} attention_type=3

.PHONY: santacoder-original
santacoder-original:
	${RUN_HF} --pretrained_model=bigcode/santacoder --tokenizer=bigcode/santacoder --trust_remote_code ${EXP_ARGS}

.PHONY: santacoder
santacoder:
	${RUN_HF} --pretrained_model=bigcode/santacoder-fast-inference --tokenizer=bigcode/santacoder ${EXP_ARGS}

.PHONY: optimized-santacoder
optimized-santacoder:
	${RUN_HF} --pretrained_model=olivierdehaene/optimized-santacoder --tokenizer=bigcode/santacoder --trust_remote_code ${EXP_ARGS}

.PHONY: santacoder-cpu
santacoder-cpu:
	python -m src.main --pipeline_class=HF_Pipeline --pretrained_model=mayank31398/santacoder --tokenizer=mayank31398/santacoder --dtype=${DTYPE} --batch_size=${BATCH_SIZE} --max_input_length=${MAX_INPUT_LENGTH} --trust_remote_code --device cpu

.PHONY: santacoder-onnx
santacoder-onnx:
	python -m src.main --pipeline_class=ONNX_Pipeline --pretrained_model=onnx-santacoder --tokenizer=onnx-santacoder --dtype=${DTYPE} --batch_size=${BATCH_SIZE} --max_input_length=${MAX_INPUT_LENGTH} --trust_remote_code --device cpu
