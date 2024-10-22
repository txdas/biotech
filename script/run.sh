#!/bin/bash
#set -exo
SDIR=/data/data/sequencing/SeV_3gene/sequencing/240924-8bin/00.mergeRawFq
TDIR=/data/home/jinyalong/data/sev_3gene/fasta
#names=(Bmix-1 Bmix-2 Gmix-1 Gmix-2 Rmix-1 Rmix-2)
names=(Gmix-1 Gmix-2)
for i in ${names};do
  {
    pear -f $SDIR/$i/${i}_raw_1.fq.gz -r $SDIR/$i/${i}_raw_2.fq.gz -o ${TDIR}/$i
    seqkit fq2fa ${TDIR}/$i.assembled.fastq -o ${TDIR}/$i.fasta
  } &
done
wait