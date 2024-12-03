import numpy as np


class Tokenizer(object):

    def __init__(self, chars="acgt"):
        self.length = len(chars)
        self.chars = chars + "n"
        self.dchars = {c: [0] * self.length for c in self.chars}
        for i, c in enumerate(chars):
            self.dchars[c][i] = 1

    def corpus(self, i):
        idx = i.argmax()
        return self.chars[idx]

    def decode(self, X):
        mask = np.max(X, axis=-1) == 0
        X = np.argmax(X, axis=-1)
        X = np.where(mask, -1, X)
        seqs = []
        for x in X:
            seq = "".join([self.chars[c] for c in x])
            seqs.append(seq)
        return seqs

    def encode(self, seqs, seq_len=None):
        vectors = np.empty([len(seqs), seq_len, self.length])
        for i, seq in enumerate(seqs):
            seq = seq[:seq_len].lower()
            lst = [self.dchars[x] for x in seq]
            if seq_len > len(seq):
                lst += [self.dchars['n']] * (seq_len - len(seq))
            a = np.array(lst)
            vectors[i] = a
        return vectors

