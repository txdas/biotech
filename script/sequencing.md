# 1. 安装驱动
```bash
BASE_DIR="/Users/john/git/UTR/data/prompt_sev"
fwd_fastq="${BASE_DIR}/15-8-25-1_raw_1.fq.gz"
rev_fastq="${BASE_DIR}/15-8-25-2_raw_1.fq.gz"
merge_fasta="${BASE_DIR}/sev240222_v1.fasta"
merge_fastq="${BASE_DIR}/sev240222_v1.fastq"

# 1. 利用pandaseq处理测序数据
conda install pandaseq -c bioconda
pandaseq -f ${fwd_fastq} -r ${rev_fastq} -w ${merge_fasta} -g log.txt
pandaseq -f DNA_raw_1.fq.gz -r DNA_raw_2.fq.gz -w DNA_merge.fq -g log.txt

# 2. 利用pear和seqkit处理测序数据
conda install pear seqkit -c bioconda
pear -f ${fwd_fastq} -r ${rev_fastq} -p 0.001 -j 20 -n 110 -o ${merge_fastq}
pear -f DNA_raw_1.fq.gz -r DNA_raw_2.fq.gz  -o ~/data/DNA_merge.fq
nohup pear -f pl3-1-1_raw_1.fq.gz -r pl3-1-1_raw_2.fq.gz  -o ~/data/pl3-1-1_merge.fq > ~/data/pl3-1-1.log 2>&1  &
nohup pear -f pl3-1-2_raw_1.fq.gz -r pl3-1-2_raw_2.fq.gz  -o ~/data/pl3-1-2_merge.fq > ~/data/pl3-1-2.log 2>&1  &

# 用于pear输出的fastq格式，需要seqkit进一步处理测序数据
seqkit fq2fa ${merge_fastq} -o ${merge_fasta}
seqkit fq2fa ~/data/DNA_merge.fq.assembled.fastq -o ~/data/DNA_merge.fasta
seqkit fq2fa ~/data/pl3-1-1_merge.fq.assembled.fastq -o ~/data/pl3-1-1_merge.fasta
seqkit fq2fa ~/data/pl3-1-2_merge.fq.assembled.fastq -o ~/data/pl3-1-2_merge.fasta
```