#!/bin/bash
set -exo
SDIR=/data/data/sequencing/SeV_3gene/sequencing/240924-8bin/00.mergeRawFq
TDIR=/data/home/jinyalong/data/sev_3gene/fasta
names=(Bmix-1 Bmix-2 Gmix-1 Gmix-2 Rmix-1 Rmix-2)
start=$(date +"%s")
for i in ${names};do
  {
    sleep 3
#    pear -f $SDIR/$i/${i}_raw_1.fq.gz -r $SDIR/$i/${i}_raw_2.fq.gz -o ${TDIR}/$i
  } &
done
wait    #等待上述程序结束
end=$(date +"%s")
echo "time: $((end - start))"