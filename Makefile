check_dirs := src scripts

style:
	black --preview $(check_dirs)
	isort $(check_dirs)

batch_size := 1

install:
	git submodule update --init
	pip install -r requirements.txt

# BLOOM AliBi
hf-1b-bloom-fp32:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class BLOOM --dtype float32 --batch_size ${batch_size}

hf-1b-bloom-bf16:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class BLOOM --dtype bfloat16 --batch_size ${batch_size}

hf-1b-bloom-int8:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class BLOOM --dtype int8 --batch_size ${batch_size}

ds-inference-1b-bloom-fp16:
	deepspeed --num_gpus 1 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class DS_Pipeline --model_class BLOOM --batch_size ${batch_size}

# GPT2 MHA
hf-1b-GPT2-mha-fp32:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype float32 --batch_size ${batch_size}

hf-1b-GPT2-mha-bf16:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype bfloat16 --batch_size ${batch_size}

hf-1b-GPT2-mha-int8:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype int8 --batch_size ${batch_size}

ds-inference-1b-GPT2-mha-fp16:
	deepspeed --num_gpus 1 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class DS_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --batch_size ${batch_size}

# GPT2 MQA1
hf-1b-GPT2-mqa1-fp32:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype float32 --batch_size ${batch_size}

hf-1b-GPT2-mqa1-bf16:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype bfloat16 --batch_size ${batch_size}

hf-1b-GPT2-mqa1-int8:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype int8 --batch_size ${batch_size}

# Input length experiments
hf-1b-GPT2-mqa1-int8-input-length:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype int8 --batch_size ${batch_size} --max_input_length ${max_input_length}

hf-1b-GPT2-mha-int8-input-length:
	python3 src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype int8 --batch_size ${batch_size} --max_input_length ${max_input_length}
