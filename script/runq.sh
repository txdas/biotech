#!/bin/bash
set -exo
SDIR=/data/data/sequencing/SeV_3gene/sequencing/240924-8bin/00.mergeRawFq
TDIR=/data/home/jinyalong/data/sev_3gene/fasta
#names=(Bmix-1 Bmix-2 Gmix-1 Gmix-2 Rmix-1 Rmix-2)
names=(Bmix-1 Gmix-2)
for i in ${names};do
  {
    pandaseq -f $SDIR/$i/${i}_raw_1.fq.gz -r $SDIR/$i/${i}_raw_2.fq.gz -p CTCGGCTTCACCGTCACC -q AGGAGATGACTCTAGTGTCAGC
  }
done
