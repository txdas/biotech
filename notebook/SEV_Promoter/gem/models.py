import torch
from torch import nn
from torchinfo import summary


def get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right= kernel_size - 1 - left
    return [ max(0,x) for x in [left,right] ]


class CNN(nn.Module):
    def __init__(self ,kernel_size=5, seq_len=44):
        super(CNN, self).__init__()
        self.seq_len =seq_len
        self.kernel_size =kernel_size
        self.padding = get_padding(kernel_size)
        self.conv0 = nn.Conv1d(in_channels=4, out_channels=100, kernel_size=(kernel_size,), padding="same")
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(kernel_size,), padding="same")
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=(kernel_size,), padding="same")
        self.conv3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=(kernel_size,), padding="same")
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        # self.bn = nn.BatchNorm1d(200)
        self.bn = nn.LayerNorm(self.seq_len//4)
        self.linear = nn.Linear(200 *(self.seq_len//4) ,100)
        self.output = nn.Linear(100 ,1)

    def forward(self, x):
        bs = x.shape[0]
        x= torch.permute(x,(0,2,1))
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.dropout(self.pool(x))
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = self.act(self.bn(self.conv3(x)))
        x = self.dropout(x).reshape((bs,-1))
        x = self.dropout(self.act(self.linear(x)))
        x = self.output(x)
        return x

    def copy(self):
        import copy
        model = CNN(self.seq_len)
        return copy.deepcopy(model)


if __name__ == '__main__':
    seq_len= 44
    model = CNN(seq_len=seq_len, kernel_size=4)
    print(summary(model, input_size=(16, seq_len, 4)))
    data = torch.rand((25, seq_len, 4)).to("cuda:0")
    print(model(data).shape)