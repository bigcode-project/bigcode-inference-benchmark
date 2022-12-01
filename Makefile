hf:
	python src/main.py --hidden_size 6144 --n_head 32 --n_layer 30 --pipeline_class HF_GPU_Pipeline --model_class GPT2 --n_positions 2048 --attention_type 2

ds-inference:
	deepspeed --num_gpus 1 src/main.py --hidden_size 6144 --n_head 32 --n_layer 30 --pipeline_class DS_Inference_Pipeline --model_class BLOOM
