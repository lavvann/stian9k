#!usr/bin/python3.6
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, CuDNNLSTM
import data
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def nn_gen(df, target, time_steps=200, epochs=50):

    arr = df.values

    train = TimeseriesGenerator(arr, target.values, length=1, sampling_rate=1, stride=1,
                                start_index=0, end_index=int(len(df.index) * 0.8),
                                shuffle=True, reverse=False, batch_size=time_steps)

    test = TimeseriesGenerator(arr, target.values, length=1, sampling_rate=1, stride=1,
                               start_index=int(len(df.index) * 0.8), end_index=None,
                               shuffle=False, reverse=False, batch_size=time_steps)

    model = Sequential()
    model.add(CuDNNLSTM(units=50, input_shape=(1, 3), return_sequences=True))
    model.add(CuDNNLSTM(units=50, return_sequences=True))
    model.add(CuDNNLSTM(units=50, return_sequences=True))
    model.add(CuDNNLSTM(units=50))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=["accuracy"])
    history = model.fit_generator(train, epochs=epochs)
    plt.plot(history.history['loss'])
    plt.show()
    #  print(history.history.keys())    # print history index keys
    print(model.summary())
    print(model.evaluate_generator(test))
    # testPredict = model.predict_generator(x_test)