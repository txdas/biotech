import numpy as np
import pandas as pd


def to_motif(ppm):
    entropy = np.zeros(ppm.shape)
    entropy[ppm > 0] = ppm[ppm > 0] * -np.log2(ppm[ppm > 0])
    entropy = np.sum(entropy, axis=1)
    conservation = 2 - entropy
    probs = np.multiply(ppm, conservation.reshape((-1, 1)))
    return pd.DataFrame(probs, columns=['A', 'C', 'G', 'T'])


def draw_motif(ldf):
    import matplotlib.pyplot as plt
    import logomaker
    f, axs = plt.subplots(figsize=(16, 3))
    logomaker.Logo(pwm, ax=axs)
    plt.show()


class Motif(object):

    def __init__(self, onehots, bg_pfm=None, weight=None, smooth_value=1e-10):
        self.onehots = onehots
        self.shape = self.onehots.shape[-2:]
        self.bg_pfm = bg_pfm if bg_pfm is not None else onehots.sum(axis=0) if weight else np.full(self.shape, 0.25)
        self.weight = weight.reshape((-1, 1, 1)) if weight is not None else np.full((onehots.shape[0], 1, 1), 1)
        self.pfm = np.multiply(self.onehots, self.weight).sum(axis=0)
        self.smooth_value = smooth_value

    @property
    def ppm(self):
        pfm = self.pfm
        return pfm / pfm.sum(axis=-1)[:, np.newaxis]

    @property
    def pwm(self):
        ppm = self.ppm
        bg_ppm = self.bg_pfm / self.bg_pfm.sum(axis=-1)[:, np.newaxis]
        return np.log2(ppm / bg_ppm)

    def kl(self):
        ppm = self.ppm
        bg_ppm = self.bg_pfm / self.bg_pfm.sum(axis=-1)[:, np.newaxis]
        pwm = np.log2(ppm / bg_ppm)
        ht_mat = (ppm * pwm).sum(axis=1).reshape((-1, 1))
        return np.multiply(ppm, ht_mat)


if __name__ == '__main__':
    onehots = np.zeror((16, 118, 4))
    scores = np.zeros(16)
    motif = Motif(onehots, weight=scores)
    pwm = motif.kl()
