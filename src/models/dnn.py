from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout,Activation,InputLayer,BatchNormalization
import numpy as np
import tensorflow as tf
import random
import warnings

# import keras
seed=1571
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
warnings.filterwarnings("ignore")

dim,UNITS,REG=1024,128,1e-4
model = Sequential()
model.add(InputLayer(batch_input_shape=(None,dim)))
model.add(Dense(units=UNITS, kernel_regularizer = l2(REG), bias_regularizer = l2(REG)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(units=1))
model.add(Activation('linear'))
#compile the model
adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss="mean_squared_error", optimizer=adam)
# model.fit(x, y, batch_size=128, epochs=nb_epoch, verbose=1)
model.summary()