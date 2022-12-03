for bs in {1,2,4,8,16,32,64}
do
    make $1 batch_size=$bs
done

for i in {1..20}
do
    bs=$(($i*128))
    make $1 batch_size=$bs
done
