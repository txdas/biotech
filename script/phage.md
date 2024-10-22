# Graphage
Graphage, a phage analysis tool that incorporates phage and ICE discrimination, phage integration site prediction, phage lifestyle prediction, and phage host prediction. Graphage utilizes a Gapped Pattern Graph Convolutional Network (GP-GCN) framework for phage representation learning. The GP-GCN framework is available for various sequence analysis at [Github](https://github.com/deepomicslab/GCNFrame).

![image](https://github.com/deepomicslab/Graphage/blob/main/Graphage.png)

# Diamond

Introduction
============

DIAMOND is a sequence aligner for protein and translated DNA searches,
designed for high performance analysis of big sequence data. The key
features are:

- Pairwise alignment of proteins and translated DNA at 100x-10,000x
    speed of BLAST.
- [Protein clustering of up to tens of billions of proteins](https://github.com/bbuchfink/diamond/wiki/Clustering)
- Frameshift alignments for long read analysis.
- Low resource requirements and suitable for running on standard
    desktops or laptops.
- Various output formats, including BLAST pairwise, tabular and XML,
    as well as taxonomic classification.

# DPProm
### DPProm: A Two-layer Predictor for Identifying Promoters and Their Types on Phage Genome Using Deep Learning
## Introduction
### Motivation:
With the number of phage genomes increasing, it is urgent to develop new bioinformatics methods for phage genome annotation. Promoter is a DNA region and important for gene transcriptional regulation. In the era of post-genomics, the availability of data made it possible to establish computational models for promoter identification with robustness.
### Results:
In this work, we proposed a two-layer model DPProm. On the first layer, for identifying the promoters, DPProm-1L was presented with a dual-channel deep neural network ensemble method fusing multi-view features, including sequence feature and handcrafted feature; on the second layer, for predicting promoter types (host or phage), DPProm-2L was proposed based on convolutional neural network (CNN). At the whole phage genome level, a novel sequence data processing workflow composed of sliding window module and merging sequences module was raised. Combined with the novel data processing workflow, DPProm could effectively decrease the false positives for promoter prediction on the whole phage genome.

# Prokka: rapid prokaryotic genome annotation

## Introduction

Whole genome annotation is the process of identifying features of interest
in a set of genomic DNA sequences, and labelling them with useful
information. Prokka is a software tool to annotate bacterial, archaeal and
viral genomes quickly and produce standards-compliant output files.

## Installation

### Bioconda
If you use [Conda](https://conda.io/docs/install/quick.html)
you can use the [Bioconda channel](https://bioconda.github.io/):
```
conda install -c conda-forge -c bioconda -c defaults prokka
saher_file: root_dir
jupyer: notebook
