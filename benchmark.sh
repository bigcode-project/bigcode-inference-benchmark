# HF
python src/hf.py --hidden_size 6144 --n_head 32 --n_layer 30

# DS-inference
deepspeed --num_gpus 1 src/ds_inference.py --hidden_size 6144 --n_head 32 --n_layer 30
