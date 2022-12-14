export CUDA_VISIBLE_DEVICES=0

rm -rf ./tmp

for max_input_length in {4,8,16,32,64,128,256,512,1024,1536,1900}
do
    make $1 batch_size=32 max_input_length=$max_input_length
done
