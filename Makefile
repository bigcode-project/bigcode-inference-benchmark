check_dirs := src scripts

style:
	black --preview $(check_dirs)
	isort $(check_dirs)

batch_size := 1

install-mqa-transformers:
	git clone https://github.com/bigcode-project/transformers.git; \
	cd transformers; \
	git checkout mayank/multi_query; \
	pip install .; \
	cd ..; \
	rm -rf transformers;

# BLOOM AliBi
hf-1b-bloom-fp32:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class BLOOM --dtype float32 --batch_size ${batch_size}

hf-1b-bloom-bf16:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class BLOOM --dtype bfloat16 --batch_size ${batch_size}

hf-1b-bloom-int8:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class BLOOM --dtype int8 --batch_size ${batch_size}

ds-inference-1b-bloom-fp16:
	deepspeed --num_gpus 1 --module src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class DS_Pipeline --model_class BLOOM --batch_size ${batch_size}

# GPT2 MHA
hf-1b-GPT2-mha-fp32:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype float32 --batch_size ${batch_size}

hf-1b-GPT2-mha-bf16:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype bfloat16 --batch_size ${batch_size}

hf-1b-GPT2-mha-int8:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype int8 --batch_size ${batch_size}

ds-inference-1b-GPT2-mha-fp16:
	deepspeed --num_gpus 1 --module src/main.py --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class DS_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --batch_size ${batch_size}

# GPT2 MQA1
hf-1b-GPT2-mqa1-fp32:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype float32 --batch_size ${batch_size}

hf-1b-GPT2-mqa1-bf16:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype bfloat16 --batch_size ${batch_size}

hf-1b-GPT2-mqa1-int8:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype int8 --batch_size ${batch_size}

# Input length experiments
hf-1b-GPT2-mqa1-int8-input-length:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2 --dtype int8 --batch_size ${batch_size} --max_input_length ${max_input_length}

hf-1b-GPT2-mha-int8-input-length:
	python -m src.main --hidden_size 2048 --n_head 16 --n_layer 24 --pipeline_class HF_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 1 --dtype int8 --batch_size ${batch_size} --max_input_length ${max_input_length}
