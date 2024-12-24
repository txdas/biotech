import numpy as np
import random
import Levenshtein
import pandas as pd

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1], 'N':[0,0,0,0]}
    return np.array([ltrdict[x] for x in seq])

def ret_rand_nuc(idx):
    lst = [0,1,2,3]
    # lst.remove(idx)
    x = random.sample(lst,1)[0]
    if x == 0:
        return [1,0,0,0] # A
    if x == 1:
        return [0,1,0,0] # C
    if x == 2:
        return [0,0,1,0] # G
    if x == 3:
        return [0,0,0,1] # T

def vector_to_nuc(arr, seq_len=24):
    seq = ''
    for i in range(seq_len):
        if arr[i,0] == 1:
            seq = seq + 'A'
        if arr[i,1] == 1:
            seq = seq + 'C'
        if arr[i,2] == 1:
            seq = seq + 'G'
        if arr[i,3] == 1:
            seq = seq + 'T'
    return seq

def simple_mutate(seq, nbr_bases=2,seq_len=24):
    lst = list(range(seq_len))
    poss = random.sample(lst,nbr_bases)
    for pos in poss:
        idx = np.argmax(seq[pos])
        seq[pos] = ret_rand_nuc(idx)
    return seq


def get_vee_seqs(nbr_bases=3, size=400):
    # VEE
    wt = 'atgggcggcgcatgagagaagcccagaccaattacctacccaaa'.upper()
    wt_vect = vectorizeSequence(wt)
    seqs1 = [ wt[:i] +j+wt[i+1:]for i in range(len(wt)) for j in "ACGT" if wt[i] != j]
    print(len(seqs1))
    seqs = [vector_to_nuc(simple_mutate(wt_vect.copy(),nbr_bases=nbr_bases,seq_len=44),seq_len=44) for _ in range(size)]
    seqs = list(set(seqs1+seqs))
    distances = [Levenshtein.distance(s,wt) for s in seqs]
    df = pd.DataFrame(data={"seq":seqs,"distance":distances})
    df.loc[len(df)]=[wt,0]
    df = df.sample(frac=1.0)
    print(df.distance.value_counts())
    return df

if __name__ == '__main__':
    get_vee_seqs()

