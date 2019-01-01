#!usr/bin/python3.6
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM
import data
import numpy as np
import pandas as pd
from datetime import datetime


def nn_gen(df, long=True, time_steps=100):
    # - df normalization
    print("Normalizing X \n")
    df['date_time'] = pd.to_timedelta(df['date_time']).dt.total_seconds().astype(int)   # convert timestamp to float
    df = data.normalize_data(df)

    print("Creating sequences for NN \n")
    if long:
        targets = df.iloc[:, 3]  # Buy signal target
    else:
        targets = df.iloc[:, 4]  # Short signal target

    df.drop('y1', axis=1, inplace=True)
    df.drop('y2', axis=1, inplace=True)

    arr = df.values

    train = TimeseriesGenerator(arr, targets.values, length=1, sampling_rate=1, stride=1,
                                start_index=0, end_index=int(len(df.index) * 0.8),
                                shuffle=True, reverse=False, batch_size=time_steps)

    test = TimeseriesGenerator(arr, targets, length=1, sampling_rate=1, stride=1,
                               start_index=int(len(df.index) * 0.8), end_index=None,
                               shuffle=True, reverse=False, batch_size=time_steps)

    model = Sequential()
    model.add(LSTM(4, input_shape=(1, 3)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    history = model.fit_generator(train, epochs=10)
    print(model.summary())
    print(str(history))
    # model.evaluate_generator(x_test)
    # testPredict = model.predict_generator(x_test)