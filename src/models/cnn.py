
from tools import one_hot_encode
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GRU, LSTM,  Bidirectional, Dense,Dropout,Activation,Flatten,Conv1D,InputLayer,Convolution1D,MaxPooling1D,BatchNormalization,Concatenate,Input



def huber_loss(y_true, y_pred):
    d = 0.15
    x = K.abs(y_true - y_pred)
    d_t = d * K.ones_like(x)
    quad = K.min(K.stack([x, d_t], axis=-1), axis=-1)
    return (0.5 * K.square(quad) + d * (x - quad))


def train_model(x, y, border_mode='same', inp_len=50, layers=3, kernel_size=8, filters=120,
                dropout=0, epochs=3, learning_rate=0.001):
    ''' Build model archicture and fit.'''
    model = Sequential()
    model.add(Input(shape=(inp_len, x.shape[-1])))
    for i in range(layers):
        model.add(Conv1D(activation="relu", padding=border_mode, filters=filters,
                         kernel_size=kernel_size))
        model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(filters))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('linear'))

    # compile the model
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="mean_squared_error", optimizer=adam)

    model.fit(x, y, batch_size=128, epochs=epochs, verbose=1)
    return model


def train_model_gru(x, y, inp_len=40, hidden_size=128, epochs=3, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(inp_len, x.shape[-1])))
    # model.add(Bidirectional(GRU(hidden_size, return_sequences=True)))
    # model.add(Bidirectional(GRU(hidden_size)))
    model.add(Bidirectional(LSTM(hidden_size)))
    model.add(Dense(hidden_size))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    # compile the model
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="mean_squared_error", optimizer=adam)

    model.fit(x, y, batch_size=128, epochs=epochs, verbose=1)
    return model



def train_bimodel(xi1, xi2, y, inp_len1=50, inp_len2=50, layers=3, kernel_size=8, filters=120,
                   epochs=3, learning_rate=0.001):
    ''' Build model archicture and fit.'''
    input1 = Input(shape=(inp_len1, xi1.shape[-1]))
    input2 = Input(shape=(inp_len2, xi2.shape[-1]))
    for i in range(layers):
        if i == 0:
            x1 = Conv1D(activation="relu", padding='same', filters=filters,
                        kernel_size=kernel_size)(input1)
            x2 = Conv1D(activation="relu", padding='same', filters=filters,
                        kernel_size=kernel_size)(input2)
        else:
            x1 = Conv1D(activation="relu", padding='same', filters=filters,
                        kernel_size=kernel_size)(x1)
            x2 = Conv1D(activation="relu", padding='same', filters=filters,
                        kernel_size=kernel_size)(x2)
    x1 = Flatten()(x1)
    x1 = Dense(filters)(x1)
    x2 = Flatten()(x2)
    x2 = Dense(filters)(x2)
    x = Concatenate()([x1, x2])
    x = Dense(filters, activation="relu")(x)
    out_put = Dense(1, activation="linear")(x)
    model = Model(inputs=[input1, input2], outputs=out_put)
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="mean_squared_error", optimizer=adam)
    model.fit([xi1, xi2], y, batch_size=128, epochs=epochs, verbose=1)
    return model


def train_normodel(x, y, inp_len=50,  layers=3, kernel_size=8, filters=120, epochs=3, learning_rate=0.001):
    REG = 1e-4
    ''' Build model archicture and fit.'''

    def add_conv(model):
        model.add(Convolution1D(filters=filters, kernel_size=kernel_size, activation=None, strides=1, padding='same',
                                kernel_regularizer=l2(REG), bias_regularizer=l2(REG)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        return (model)

    model = Sequential()
    input_shape = (inp_len, 4)
    model.add(InputLayer(batch_input_shape=(None,) + input_shape))
    for i in range(layers):
        model = add_conv(model)
    model.add(Flatten())
    model.add(Dense(units=filters, kernel_regularizer=l2(REG), bias_regularizer=l2(REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=filters, kernel_regularizer=l2(REG), bias_regularizer=l2(REG)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    # compile the model
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=huber_loss, optimizer=adam)
    model.fit(x, y, batch_size=128, epochs=epochs, verbose=1)
    return model
