for max_input_length in {4,8,16,32,64,128,256,512,1024,1536,1900}
do
    make $1 batch_size=32 MAX_INPUT_LENGTH=$max_input_length
done
