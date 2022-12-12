export CUDA_VISIBLE_DEVICES=0

rm -rf ./tmp

# split for loops
for i in {0..20}
do
    bs=$((2**$i))
    echo $bs
    # make $1 batch_size=$bs
done
