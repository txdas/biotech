import pandas as pd
from tools import one_hot_encode, r2, test_data, split_polya, test_bidata
from sklearn import preprocessing
from cnn import train_model, train_normodel, train_bimodel, train_model_gru
import scipy.stats as stats


def load_data():
    df = pd.read_csv("/Users/john/git/UTR/data/prompt_sev/158252c.csv")
    # df = pd.read_csv("/Users/john/git/UTR/data/prompt_sev/158252.csv")
    df = df.dropna()
    df[["seq1", "seq2"]] = df.apply(lambda x: split_polya(x["seq"]), axis=1, result_type="expand")
    plasmid_gate, rna_gate, seq_len = 50, 0, 45
    df = df[(df["rna_counts"] > rna_gate) & (df["plasmid_counts"] > plasmid_gate)]
    e_train = df.sample(frac=0.9)
    e_test = df[~df.index.isin(e_train.index)]
    print(e_train.shape, e_test.shape)
    # Scale the training mean ribosome load values
    e_train.loc[:, 'scaled_rl'] = preprocessing.StandardScaler().fit_transform(
        e_train.loc[:, 'score'].values.reshape(-1, 1))
    return e_train, e_test, seq_len


def train(e_train, e_test, seq_len=40, epochs=10):
    seq_e_test = one_hot_encode(e_test["seq"], seq_len=seq_len)
    seq_e_train = one_hot_encode(e_train["seq"], seq_len=seq_len)
    # model = train_model(seq_e_train, e_train['scaled_rl'], epochs=10, inp_len=seq_len, layers=2,
    #                     filters=128, learning_rate=0.001)
    # model = train_normodel(seq_e_train, e_train['scaled_rl'], epochs=10, inp_len=seq_len, layers=2,
    #                     filters=100, learning_rate=0.001)
    model = train_model_gru(seq_e_train, e_train['scaled_rl'], epochs=epochs, inp_len=seq_len)
    e_test = test_data(df=e_test, model=model, obs_col='score', test_seq=seq_e_test)
    r = r2(e_test['score'], e_test['pred'])
    pr = stats.pearsonr(e_test['score'], e_test['pred'])
    print('test r-squared = ', r, "pearsonR = ", pr[0])
    e_train = test_data(df=e_train, model=model, obs_col='score', test_seq=seq_e_train)
    r = r2(e_train['score'], e_train['pred'])
    pr = stats.pearsonr(e_train['score'], e_train['pred'])
    print('train r-squared = ', r, "pearsonR = ", pr[0])


def trainbi(e_train, e_test, inp_len1=78, inp_len2=40):
    seq1_train = one_hot_encode(e_train["seq1"],seq_len=inp_len1)
    seq2_train = one_hot_encode(e_train["seq2"],seq_len=inp_len2)
    seq1_test = one_hot_encode(e_test["seq1"],seq_len=inp_len1)
    seq2_test = one_hot_encode(e_test["seq2"],seq_len=inp_len2)
    model = train_bimodel(seq1_train, seq2_train, e_train['scaled_rl'], epochs=10, inp_len1=78, inp_len2=40, layers=2,
                        filters=100, learning_rate=0.001)
    e_test = test_bidata(df=e_test, model=model, obs_col='score', test_seq1=seq1_test, test_seq2=seq2_test)
    r = r2(e_test['score'], e_test['pred'])
    pr = stats.pearsonr(e_test['score'], e_test['pred'])
    print('test r-squared = ', r, "pearsonR = ", pr[0])
    e_train = test_bidata(df=e_train, model=model, obs_col='score', test_seq1=seq1_train, test_seq2=seq2_train)
    r = r2(e_train['score'], e_train['pred'])
    pr = stats.pearsonr(e_train['score'], e_train['pred'])
    print('train r-squared = ', r, "pearsonR = ", pr[0])


if __name__ == '__main__':
    e_train, e_test, seq_len = load_data()
    train(e_train, e_test, seq_len, epochs=20)
    # trainbi(e_train, e_test)