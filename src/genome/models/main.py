from torchinfo import summary
from cnn import CNN, CNN_LSTM

if __name__ == '__main__':
    seq_len = 44
    # model = CNN(seq_len=seq_len)
    model = CNN_LSTM(seq_len=seq_len)
    print(summary(model, input_size=(16, seq_len, 4)))