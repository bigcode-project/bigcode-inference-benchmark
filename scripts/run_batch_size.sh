for i in {1..20}
do
    bs=$(($i*128))
    make $1 BATCH_SIZE=$bs
done
