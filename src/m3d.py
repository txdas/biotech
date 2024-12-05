import os.path

import pandas as pd
import numpy as np
import torch
import matplotlib as mpl
import umap
from torch import nn, optim,utils
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler



class CNN(nn.Module):

    def __init__(self, seq_len=44):
        super(CNN, self).__init__()
        self.seq_len = seq_len
        self.model = self.cnn1_model()

    def cnn1_model(self, ):
        model = nn.Sequential()
        model.append(nn.Conv1d(in_channels=4, out_channels=100, kernel_size=(5,), padding=2))
        model.add_module("conv0", nn.ReLU())
        model.append(nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(5,), padding=2))
        model.add_module("conv1", nn.ReLU())
        model.append(nn.MaxPool1d(2, stride=2))
        model.append(nn.Dropout(p=0.3))
        model.append(nn.Conv1d(in_channels=100, out_channels=200, kernel_size=(5,), padding=2))
        model.add_module("conv2", nn.ReLU())
        model.append(nn.MaxPool1d(2, stride=2))
        model.append(nn.Conv1d(in_channels=200, out_channels=200, kernel_size=(5,), padding=2))
        model.append(nn.BatchNorm1d(200))
        model.add_module("conv3", nn.ReLU())
        model.append(nn.Dropout(p=0.3))
        model.append(nn.Flatten())
        model.add_module("linear", nn.Linear(200 * (seq_len // 4), 100))
        model.append(nn.ReLU())
        model.append(nn.Dropout(p=0.3))
        model.append(nn.Linear(100, 1))
        return model


    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        return self.model(x)


class WrapModel(object):
    def __init__(self, model):
        self.model = model

    def extract_layer_output(self, x, layer_name):
        buffer = []

        def regist_hook():
            def layer_hook(module, inp, out):
                buffer.append(out.data.detach().numpy())

            getattr(self.model.model, layer_name).register_forward_hook(layer_hook)

        regist_hook()
        self.model.eval()
        y = self.model(x)
        layer_out = np.concatenate(buffer)
        if len(layer_out.shape) == 3:
            layer_out = np.max(layer_out, axis=1)
        return layer_out, y.detach().numpy()


def one_hot_encode(df, col='seq', seq_len=44):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        lst = [nuc_d[x] for x in seq]
        if seq_len > len(seq):
            lst += [nuc_d['n']] * (seq_len - len(seq))
        a = np.array(lst)
        vectors[i] = a
    return vectors

# name,seq_len,suff="INFL_0522",26,"" # 0.94,0.94,0.95
# wt = "AGCAAAAGCAGGGTGACAAAAACATA" # INFL

name,seq_len,suff="VEE_0816",44,"_tol_seq" # 0.83,0.83,0.82
wt = "atgggcggcgcatgagagaagcccagaccaattacctacccaaa".upper() # INFL

fdata = "C:/Users/jinya/Desktop/bio"
datadir=f"{fdata}/5UTR/{name}_"
e_train = pd.read_csv(f"{datadir}train{suff}.csv")[["seq"]]
e_test= pd.read_csv(f"{datadir}test{suff}.csv")[["seq"]]
e_wt = pd.DataFrame({"seq":[wt]})
e_train = pd.concat((e_train,e_wt))
e_test= pd.concat((e_train,e_wt))

train_x = one_hot_encode(e_train,seq_len=seq_len)
test_x = one_hot_encode(e_test, seq_len=seq_len)
mpl.rcParams['figure.dpi'] = 120
pdf,input_x = e_test, test_x
# pdf,input_x = e_train, train_x
mname = name.split("_")[0].lower()
<<<<<<< HEAD
model_dir = os.path.abspath(f"{fdata}/models/{name}.pt")
model = torch.load(model_dir)
wmodel = WrapModel(model)
layer_out, pred=wmodel.extract_layer_output(torch.Tensor(input_x),"conv1") # linear, conv1 conv2 conv3
=======
model = torch.load(f"{fdata}/models/{name}.pt")
wmodel=WrapModel(model)
layer_out, pred=wmodel.extract_layer_output(torch.Tensor(input_x),"conv2") # linear, conv1 conv2 conv3
>>>>>>> 755d5336d6a5ec86eb7538d24a5a4d80822fa8af
current = 0
grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
scores = pred.flatten()
<<<<<<< HEAD
reducer = umap.UMAP(low_memory=True,n_neighbors=20, min_dist=0.5, n_components=2, metric="euclidean")
=======
reducer = umap.UMAP(densmap=True,low_memory=True,n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean")
>>>>>>> 755d5336d6a5ec86eb7538d24a5a4d80822fa8af
conv_output_reduce = reducer.fit_transform(layer_out)
conv_output_reduce = MinMaxScaler().fit_transform(conv_output_reduce)
index = pdf[pdf.seq == wt].index[0]
highlight_point = conv_output_reduce[index]
highlight_score = scores[index]
grid_data = griddata(conv_output_reduce, scores, (grid_x, grid_y), method="cubic") # linear, cubic, nearest
print(current, max(scores), min(scores), np.nan_to_num(grid_data).max(),np.nan_to_num(grid_data).min())
fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.set_axis_off()
surface =ax.plot_surface(grid_x, grid_y,  grid_data, cmap='viridis',linewidth=0, antialiased=False, alpha=0.5)
ax.scatter(highlight_point[0], highlight_point[1], highlight_score, color="black", s=50, label="Highlighted Sequence",
           edgecolors='k')
fig.colorbar(surface, ax = ax, shrink = 0.5, aspect = 5)
ax.set_title(f"min_dist")
plt.show()
