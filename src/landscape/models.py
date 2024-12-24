import torch
from torch import nn
import os
import numpy as np

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
            layer_out = np.max(layer_out[:,:2,:], axis=1)
        return layer_out, y.detach().numpy()

def load_vee_model():
    fdata = "C:/Users/jinya/Desktop/bio"
    name = "VEE_0816"
    model_dir = os.path.abspath(f"{fdata}/models/{name}.pt")
    model = torch.load(model_dir,weights_only=False)
    wmodel = WrapModel(model)
    return wmodel