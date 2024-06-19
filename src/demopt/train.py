import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.stats as stats
import pandas as pd
from sklearn import preprocessing
from dataset import VEE5UTRDataset
from torch.utils.data import DataLoader
import torch
import random
import tqdm
from colab.utils import LRScheduler, EarlyStopping
from models import LitCNN, MyPrintingCallback, CNN


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)


def evaluate(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1, 1))
    model.eval()
    # Make predictions
    test_seq = torch.tensor(test_seq, dtype=torch.float)
    predictions = model(test_seq).reshape(-1, 1).detach().numpy()
    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df


def one_hot_encode(df, col='seq', seq_len=44):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a' :[1 ,0 ,0 ,0] ,'c' :[0 ,1 ,0 ,0] ,'g' :[0 ,0 ,1 ,0] ,'t' :[0 ,0 ,0 ,1], 'n' :[0 ,0 ,0 ,0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors


def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def foo():
    setup_seed(1337)
    # df, seq_len, epochs = pd.read_csv("./data/VEE_5UTR_0429/VEE-0429.csv"), 44, 5
    # df, seq_len, epochs = pd.read_csv("./data/VEE_3UTR/VEE_3UTR.csv"), 118, 20
    # plasmid_gate, rna_gate = 20, 4
    df = pd.read_csv("./data/VEE_5UTR_0611/VEE-0611.csv")
    plasmid_gate, rna_gate, seq_len, epochs = 30, 5, 44, 8
    # e_train = pd.read_csv("/Users/john/git/UTR/data/prompt_sev/118_rna2_240222_train.csv")
    # e_test = pd.read_csv("/Users/john/git/UTR/data/prompt_sev/118_rna2_240222_test.csv")
    # df = pd.concat([e_train, e_test], axis=0)
    # plasmid_gate, rna_gate, seq_len, epochs = 50, 10, 118, 5
    # df["seq"] = df["seq"].apply(lambda x: x[49:74])
    # seq_len = 25
    df = df[(df["rna_counts"] > rna_gate) & (df["plasmid_counts"] > plasmid_gate)]
    e_train = df.sample(frac=0.9)
    e_test = df[~df.index.isin(e_train.index)]
    seq_e_train = one_hot_encode(e_train, seq_len=seq_len)
    seq_e_test = one_hot_encode(e_test, seq_len=seq_len)
    # batch_size, filters, kernel_size, lr, hidden_size = 128, 120, 8, 1e-3, 40  # 5utr 0.8143-->0.8239, 3utr 0.7330
    # 3UTR PARAM 0.7214
    # batch_size, filters, kernel_size, lr, hidden_size = 64, 100, 4, 4.6e-4, 40
    # 5UTR PARAM  0.8073 -->0.8132
    batch_size, filters, kernel_size, lr, hidden_size = 256, 120, 4, 1e-3, 40
    # Prompt sev
    # batch_size, filters, kernel_size, lr, hidden_size = 64, 100, 8, 1.5e-5, 60
    # Scale the training mean ribosome load values
    e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(
        e_train.loc[:, 'score'].values.reshape(-1, 1))
    e_test.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(
        e_test.loc[:, 'score'].values.reshape(-1, 1))
    train_dt = VEE5UTRDataset(seq_e_train, e_train["scaled_rl"])
    test_dt = VEE5UTRDataset(seq_e_test, e_test["scaled_rl"])
    train_loader = DataLoader(train_dt, batch_size=batch_size)
    test_loader = DataLoader(test_dt, batch_size=batch_size)
    model = CNN(input_size=seq_len, filters=filters, kernel_size=kernel_size, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    criterion = torch.nn.MSELoss()
    valid_epochs_loss = []
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr*0.01)
    early_stopping = EarlyStopping(patience=3)
    for epoch in range(epochs):
        model.train()
        valid_epoch_loss = []
        for idx, (data_x, data_y) in tqdm.tqdm(enumerate(train_loader),total=len(train_loader)):
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)
            outputs = model(data_x)
            optimizer.zero_grad()
            loss = criterion(data_y, outputs)
            loss.backward()
            optimizer.step()
        model.eval()
        for idx, (data_x, data_y) in enumerate(test_loader):
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)
            outputs = model(data_x)
            loss = criterion(data_y, outputs)
            valid_epoch_loss.append(loss.item())
        valid_loss = np.average(valid_epoch_loss)
        valid_epochs_loss.append(valid_loss)
        lr_scheduler.step()
        print("epoch={}/{} of train, lr={}, valid loss={}".format(epoch + 1, epochs, lr_scheduler.get_last_lr(),  valid_loss))
        early_stopping(valid_loss)
        if early_stopping.early_stop:
            break
    model.eval()
    e_test = evaluate(e_test, model, seq_e_test, 'score', output_col='pred')
    r = r2(e_test['score'], e_test['pred'])
    print('r-squared = ', r)


if __name__ == '__main__':
    # main()
    foo()