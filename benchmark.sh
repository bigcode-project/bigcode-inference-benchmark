# HF
python src/main.py --hidden_size 6144 --n_head 32 --n_layer 30 --pipeline_class HF_GPU_Pipeline

# DS-inference
deepspeed --num_gpus 1 src/main.py --hidden_size 6144 --n_head 32 --n_layer 30 --pipeline_class DS_Inference_Pipeline
