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
