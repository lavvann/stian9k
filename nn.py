#!usr/bin/python3.6
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, CuDNNLSTM
from keras import optimizers
import data
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as k


def nn_gen(df, target, *args):
    # make tensorflow not allocate all gpu memory at start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    k.tensorflow_backend.set_session(tf.Session(config=config))

    epochs = args[0]
    steps = args[1]
    lr = args[2]
    lstm_layers = args[3]
    dense_layers = args[4]
    neurons = args[5]

    # prepare data for NN
    arr = df.values

    train = TimeseriesGenerator(arr, target[:, 1], length=1, sampling_rate=1, stride=1,
                                start_index=0, end_index=int(len(df.index) * 0.8),
                                shuffle=True, reverse=False, batch_size=steps)

    test = TimeseriesGenerator(arr, target[:, 1], length=1, sampling_rate=1, stride=1,
                               start_index=int(len(df.index) * 0.8), end_index=None,
                               shuffle=False, reverse=False, batch_size=steps)

    # make NN model
    model = Sequential()
    # add input lstm layer
    model.add(CuDNNLSTM(units=neurons, input_shape=(1, 3), return_sequences=True))
    # add lstm layers
    for i in range(0, lstm_layers, 1):
        model.add(CuDNNLSTM(units=neurons, return_sequences=True))
    # add LSTM layer without return sequence to enable dense output
    model.add(CuDNNLSTM(neurons))
    # add dense layers
    for i in range(0, dense_layers, 1):
        model.add(Dense(neurons, activation='relu'))
    # add output layer
    model.add(Dense(1, activation='sigmoid'))
    # define optimizer
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
    # train model
    history = model.fit_generator(train, epochs=epochs, validation_data=test)
    plt.plot(history.history['loss'])
    print(history.history.keys())    # print history index keys
    print(model.summary())
    # evaluate model with test data
    print(model.evaluate_generator(test))
    plt.show()
    # testPredict = model.predict_generator(x_test)
