#!/usr/bin/python3.6
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, CuDNNLSTM
from keras import optimizers
import data
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.data.python.ops import sliding
from keras import backend as k
from sklearn.preprocessing import MinMaxScaler


# Parameters
STEPS = 200      # amount of timesteps in each sequence
# LR = 1e-3        # Learning rate
INTERVAL = 5     # intervall between timesteps in sequence?
LSTM_LAYERS = 50
DENSE_LAYERS = 0
NEURONS = 150
EPOCHS = 2000


# make tensorflow not allocate all gpu memory at start
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))


""" ------- MAIN --------- """
# check for input csv
if not len(sys.argv) > 1:
    exit()
df, dn, success = data.import_processed_data("", 0, INTERVAL, sys.argv[1])
if not success:
    print("Could not import file, exiting.\n")
    exit()
print(str(dn[0]) + " \n")

# Batch size, sequnce size:
BATCH_SIZE_TRAIN = round((len(dn)/STEPS)/20)
BATCH_SIZE_TEST = round(BATCH_SIZE_TRAIN/5)
END_INDEX_TRAIN = int(len(dn) * 0.8)
START_INDEX_TEST = round(len(dn)*0.8-1)

# Normalize X data:
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(dn[:, [1]])
normalized = scaler.transform(dn[:, [1]])
dn[:, 1] = normalized[:, 0]

# ---------------- TIME SERIES GENERATOR TEST ---------------------
# Try generate batches using keras timeseriesgenerator
# need to shift Y one step down to mach y to x (defaults to one time step down)
dn[:, 2] = np.roll(dn[:, 2], 1)

train = TimeseriesGenerator(dn[:, [1]], dn[:, 2], length=STEPS, sampling_rate=1, stride=1,
                            start_index=0, end_index=END_INDEX_TRAIN,
                            shuffle=False , reverse=False, batch_size=BATCH_SIZE_TRAIN)

val = TimeseriesGenerator(dn[:, [1]], dn[:, 2], length=STEPS, sampling_rate=1, stride=1,
                            start_index=START_INDEX_TEST, end_index=(len(dn)-1),
                            shuffle=False , reverse=False, batch_size=BATCH_SIZE_TEST)
x0, y0 = train[0]
x1, y1 = val[0]
# print("y0 :" + str(y0) + "\n")
# print("x1 :" + str(x1) + "\n")
# print("y1 :" + str(y1) + "\n")
print("\n\nLength of train: " + str(len(train)) + ", Shape of x: " + str(x0.shape) + ", Shape of y: " + str(y0.shape))
print("Length of val: " + str(len(val)) + ", Shape of x: " + str(x1.shape) + ", Shape of y: " + str(y1.shape) + "\n")
# make NN model
model = Sequential()
# reduction ratio neuron each layer, last added layer neurons/2
if LSTM_LAYERS: lstm_red = int((NEURONS/2)/LSTM_LAYERS)
if DENSE_LAYERS: dense_red = int((NEURONS/2)/DENSE_LAYERS)
# add input lstm layer
model.add(CuDNNLSTM(units=NEURONS, input_shape=(STEPS/INTERVAL, 1 ), return_sequences=True))
# model.add(LSTM(units=NEURONS, input_shape=(STEPS, 1 ), return_sequences=True, activation='relu'))
# add lstm layers
for i in range(0, LSTM_LAYERS, 1):
    NEURONS = NEURONS - int(lstm_red)
    model.add(CuDNNLSTM(units=NEURONS, return_sequences=True))
# add LSTM layer without return sequence to enable dense output
model.add(CuDNNLSTM(units=NEURONS))
# add dense layers
for i in range(0, DENSE_LAYERS, 1):
    NEURONS = NEURONS - (i * dense_red)
    model.add(Dense(NEURONS, kernel_initializer='he_normal', activation='relu'))
# add output layer
model.add(Dense(1))
# optimizer to use
opt = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
opt = optimizers.nadam(lr=0.001, epsilon=None)
# compile model
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.compile(loss='mse', optimizer=opt)

print(model.summary())

# start training
history = model.fit_generator(train, epochs=EPOCHS, validation_data=val)
print("\n")
# evaluate model with test data
# print(model.evaluate_generator(test))
# print("\n")
# testPredict = model.predict_generator(val)
# print(testPredict)
print("\n\nLength of train: " + str(len(train)) + ", Shape of x: " + str(x0.shape) + ", Shape of y: " + str(y0.shape))
print("Length of val: " + str(len(val)) + ", Shape of x: " + str(x1.shape) + ", Shape of y: " + str(y1.shape) + "\n")
#
# Plot training & validation accuracy values
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

exit()

