import os
BASE = os.path.dirname(__file__)
TOPICS_PMID = os.path.abspath(os.path.join(BASE, "../conf/data/topics/Topic-{}_pmid.txt"))
TOPICS_INFO = os.path.abspath(os.path.join(BASE, "../conf/data/Topic-{}_info.jsonl"))
TOPICS_PDF = os.path.abspath(os.path.join(BASE, "../conf/data/Topic-{}_pdf.jsonl"))
RAW_PMID = os.path.abspath(os.path.join(BASE, "../conf/data/zhida/pmids.txt"))
RAW_INFO = os.path.abspath(os.path.join(BASE, "../conf/data/zhida/info.jsonl"))
RAW_REFER_INFO = os.path.abspath(os.path.join(BASE, "../conf/data/zhida/refer_info.jsonl"))
RAW_PDF = os.path.abspath(os.path.join(BASE, "../conf/data/zhida/pdf.jsonl"))
RAW_REFER_PDF = os.path.abspath(os.path.join(BASE, "../conf/data/zhida/refer_pdf.jsonl"))
ALL_PMID = os.path.abspath(os.path.join(BASE, "../conf/data/all/pmids.txt"))
ALL_INFO = os.path.abspath(os.path.join(BASE, "../conf/data/all/info.jsonl"))
ALL_REFER_INFO = os.path.abspath(os.path.join(BASE, "../conf/data/all/refer_info.jsonl"))
ALL_PDF = os.path.abspath(os.path.join(BASE, "../conf/data/all/pdf.jsonl"))
ALL_REFER_PDF = os.path.abspath(os.path.join(BASE, "../conf/data/all/refer_pdf.jsonl"))
PAIRS = os.path.abspath(os.path.join(BASE, "../conf/pairs.pkl"))
PAIRS_JSONL = os.path.abspath(os.path.join(BASE, "../conf/pairs.jsonl"))
ENV_JSONL = os.path.abspath(os.path.join(BASE, "../conf/env2.jsonl"))
PRO_JSONL = os.path.abspath(os.path.join(BASE, "../conf/pro.jsonl"))
GENE_JSONL = os.path.abspath(os.path.join(BASE, "../conf/gene.jsonl"))
PRO_clean_JSONL = os.path.abspath(os.path.join(BASE, "../conf/pro_clean.jsonl"))
FINAL = os.path.abspath(os.path.join(BASE, "../conf/final.pkl"))
# PROTEINS = os.path.abspath(os.path.join(BASE, "../conf/proteins.jsonl"))
PROTEINS = os.path.abspath(os.path.join(BASE, "../conf/build/proteins.jsonl"))
UNIPROTIDS= os.path.abspath(os.path.join(BASE, "../conf/pids.txt"))
UNIPROT = os.path.abspath(os.path.join(BASE,"../conf/data/uniprot.tsv"))
HUMAN = os.path.abspath(os.path.join(BASE,"../conf/human.csv"))
PROMPTS = os.path.abspath(os.path.join(BASE,"../conf/prompts.csv"))
TARGET = os.path.abspath(os.path.join(BASE,"../conf/data/baseline.csv"))
HITS = os.path.abspath(os.path.join(BASE,"../conf/hits.csv"))
TURING = os.path.abspath(os.path.join(BASE, "../conf/turing.csv"))
BIONER = os.path.abspath(os.path.join(BASE, "../conf/bioner.jsonl"))