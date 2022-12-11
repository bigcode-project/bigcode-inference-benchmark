export CUDA_VISIBLE_DEVICES=0

rm -rf ./tmp

for bs in {1,2,4,8,16,32,64}
do
    make $1 batch_size=$bs
done

# split for loops
for i in {1..20}
do
    bs=$(($i*128))
    make $1 batch_size=$bs
done

for i in {21..40}
do
    bs=$(($i*128))
    make $1 batch_size=$bs
done

for i in {41..60}
do
    bs=$(($i*128))
    make $1 batch_size=$bs
done

for i in {61..80}
do
    bs=$(($i*128))
    make $1 batch_size=$bs
done
