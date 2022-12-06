export CUDA_VISIBLE_DEVICES=0

rm -rf ./tmp

for max_input_length in {1900,1024,512,256,128,64,32,16,8,4}
do
    make $1 batch_size=4 max_input_length=$max_input_length
done