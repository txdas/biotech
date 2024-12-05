from collections import Counter
import numpy as np


def random_mutant(original_seq):
    # bg_seqs
    seqs = [original_seq]
    cnter = Counter()
    for i in range(len(original_seq)):
        for j, nt in enumerate(["A", "C", "T", "T"]):
            if seqs[i].upper() != nt:
                seq = original_seq[:i] + nt + original_seq[i + 1:]
                seqs.append(seq)
                cnter.update(s.lower())
    total = sum(cnter.values())
    bgnts = {k: v / total for k, v in cnter.items()}
    entropy = sum(-v * np.log2(v) for v in bgnts.values())
    return seqs, bgnts, entropy
