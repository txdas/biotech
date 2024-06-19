import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
import Levenshtein


def r2(x,y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value**2


def split_polya(s):
    match = "ATTAAGAAAAACTTAGGGTGAAAGA"
    length = len(match)
    if not s or len(s)<length:
        return s, ""
    dist, idx, hit = len(s), 0, s[:length]
    for i in range(0, len(s)-length):
        tmp = Levenshtein.distance(match, s[i:i+length])
        if tmp<dist:
            dist = tmp
            idx = i
    return s[:idx+11], s[idx+11:]


def seq_core(s):
    match = "ATTAAGAAAAACTTAGGGTGAAAGA"
    length = len(match)
    if not s or len(s)<length:
        return s
    dist, idx, hit = len(s), 0, s[:length]
    for i in range(0, len(s)-length):
        tmp = Levenshtein.distance(match, s[i:i+length])
        if tmp<dist:
            dist = tmp
            idx = i
    return s[idx:idx+length]


def one_hot_encode(series, seq_len=44):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a' :[1 ,0 ,0 ,0] ,'c' :[0 ,1 ,0 ,0] ,'g' :[0 ,0 ,1 ,0] ,'t' :[0 ,0 ,0 ,1], 'n' :[0 ,0 ,0 ,0]}
    # Creat empty matrix.
    vectors =np.empty([len(series) ,seq_len ,4])
    # Iterate through UTRs and one-hot encode
    for i ,seq in enumerate(series.str[:seq_len]):
        seq = seq.lower()
        lst = [nuc_d[x] for x in seq]
        if seq_len >len(seq):
            lst += [nuc_d['n'] ] *(seq_len -len(seq))
        a = np.array(lst)
        vectors[i] = a
    return vectors


def test_data(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1, 1))

    # Make predictions
    predictions = model.predict(test_seq).reshape(-1, 1)

    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df


def test_bidata(df, model, test_seq1, test_seq2, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1, 1))

    # Make predictions
    predictions = model.predict([test_seq1, test_seq2]).reshape(-1, 1)

    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df