import torch
from torch import nn


class CNN(nn.Module):

    def __init__(self, seq_len=44):
        super().__init__()
        self.seq_len = seq_len
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=(5,), padding=2)
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(5,), padding=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=(5,), padding=2)
        self.conv3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=(5,), padding=2)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(200)
        self.linear = nn.Linear(200 * (self.seq_len // 4), 100)
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        bs = x.shape[0]
        x = torch.permute(x, (0, 2, 1))
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.dropout(self.pool(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.bn(self.conv3(x)))
        x = self.dropout(x).reshape((bs, -1))
        x = self.dropout(self.act(self.linear(x)))
        x = self.output(x)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self, seq_len=44):
        super().__init__()
        self.seq_len = seq_len
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=(5,), padding=2)
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(5,), padding=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=(5,), padding=2)
        self.conv3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=(5,), padding=2)
        self.lstm = nn.LSTM(200,200,batch_first=True,bidirectional=True)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(200)
        self.linear = nn.Linear(200 * 2, 100)
        self.output = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.dropout(self.pool(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.bn(self.conv3(x)))
        x = self.dropout(x).permute((0,2,1))
        x = self.lstm(x)[0][:,-1,:]
        x = self.dropout(self.act(self.linear(x)))
        x = self.output(x)
        return x
