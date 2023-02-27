for bs in $1
do
  for seq in $2
  do
    for tok in $3
    do
      "${@:5}" --save="$4"_bs_"$bs"_seq_"$seq"_tok_"$tok".json --batch_size=$bs --max_input_length=$seq --max_new_tokens=$tok
    done
  done
done

# PYTHONPATH=. ./scripts/run_grid.sh "1 2 4" "50 100 200" "1 50 100" ./results/mqa_small python3 src/main.py --pipeline_class=HF_Pipeline --tokenizer=bigcode/santacoder --model_type=gpt_bigcode --dtype=float32 --device=cpu --max_log_outputs=1 --cycles=1 n_positions=512 n_embd=512 n_head=8 n_inner=2048 n_layer=8
