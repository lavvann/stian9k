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
STEPS = 200     # amount of timesteps in each sequence
LR = 1e-3     # Learning rate
INTERVAL = 1    # intervall between timesteps in sequence?
STRIDE = int(STEPS/2)  # time intervall between sequences
LSTM_LAYERS = 0
DENSE_LAYERS = 0
NEURONS = 50
EPOCHS = 500


# make tensorflow not allocate all gpu memory at start
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
k.tensorflow_backend.set_session(tf.Session(config=config))


""" ------- MAIN --------- """
# check for input csv
if not len(sys.argv) > 1:
    exit()
df, dn, success = data.import_processed_data("", 0, 0, sys.argv[1])
if not success:
    print("Could not import file, exiting.\n")
    exit()
print(str(dn[0]) + " \n")

# Batch size, sequnce size:
BATCH_SIZE_TRAIN = int((len(dn)/STEPS)/2)
BATCH_SIZE_TEST = int((len(dn)/STEPS))
STEPS = STEPS*INTERVAL
END_INDEX_TRAIN = int(len(dn) * 0.8)
# END_INDEX_TRAIN = (len(dn)-1)
START_INDEX_TEST = round(len(dn)*0.8-1)
# START_INDEX_TEST = 0

# Normalize X data:
#v = dn[:, 1]
#v_max = v.max()
#v_min = v.min()
#dn[:, 1] = (v - v_min) / (v_max - v_min)    
scaler = MinMaxScaler(feature_range=(0, 1))  
scaler = scaler.fit(dn[:, [1]])
normalized = scaler.transform(dn[:, [1]])
dn[:, 1] = normalized[:, 0]
# ---------------- TIME SERIES GENERATOR TEST ---------------------
# Try generate batches using keras timeseriesgenerator
# need to shift Y one step down to mach y to x (defaults to one time step down)
dn[:, 2] = np.roll(dn[:, 2], 1)

train = TimeseriesGenerator(dn[:, [1]], dn[:, 2], length=STEPS, sampling_rate=INTERVAL, stride=STRIDE,
                            start_index=0, end_index=END_INDEX_TRAIN,
                            shuffle=False , reverse=False, batch_size=BATCH_SIZE_TRAIN)

val = TimeseriesGenerator(dn[:, [1]], dn[:, 2], length=STEPS, sampling_rate=INTERVAL, stride=STRIDE,
                            start_index=START_INDEX_TEST, end_index=(len(dn)-1),
                            shuffle=False , reverse=False, batch_size=BATCH_SIZE_TEST)
x0, y0 = train[0]
x1, y1 = val[0]
# print("x0 :" + str(x0) + "\n")
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
# model.add(CuDNNLSTM(units=NEURONS, input_shape=(STEPS, 1 ), return_sequences=True))
model.add(LSTM(units=NEURONS, input_shape=(STEPS, 1 ), return_sequences=True, activation='relu'))
# add lstm layers
for i in range(0, LSTM_LAYERS, 1):
    NEURONS = NEURONS - (i * lstm_red)
    model.add(LSTM(units=NEURONS, return_sequences=True, activation='relu'))
# add LSTM layer without return sequence to enable dense output
model.add(LSTM(units=NEURONS))
# add dense layers
for i in range(0, DENSE_LAYERS, 1):
    NEURONS = NEURONS - (i * dense_red)
    model.add(Dense(NEURONS, kernel_initializer='he_normal', activation='relu'))
# add output layer
model.add(Dense(1))
# optimizer to use
opt = optimizers.SGD(lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
# compile model
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.compile(loss='mse', optimizer='nadam')
""" print weights before training
# for i in range(0, len(model.layers), 1):
#     print(str(model.layers[i].get_weights()[1]))
# print model structure"""

print(model.summary())

# start training
history = model.fit_generator(train, epochs=EPOCHS, validation_data=val)
print("\n")
# evaluate model with test data
# print(model.evaluate_generator(test))
# print("\n")
testPredict = model.predict_generator(val)
print(testPredict)
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

"""
# --------------- DATA GENERATOR TEST -----------------------------
sequence = dn[:, 1].reshape(len(dn), 1)  #
labels = dn[:, 3]
#labels = dn[:, 3].tolist()  #

# create TensorFlow Dataset object
data = tf.data.Dataset.from_tensor_slices((sequence, labels))

# sliding window batch
window_size = 10
window_shift = 1
data = data.apply(sliding.sliding_window_batch(window_size=window_size, window_shift=window_shift))
# data = data.shuffle(1000, reshuffle_each_iteration=False)
data = data.batch(1)


# WARNING:tensorflow:From /home/stian/.local/lib/python3.6/site-packages/tensorflow/python/util/deprecation.py:488: sliding_window_batch (from tensorflow.contrib.data.python.ops.sliding) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use `tf.data.Dataset.window(size=window_size, shift=window_shift, stride=window_stride).flat_map(lambda x: x.batch(window.size))` instead.


#iter = dataset.make_initializable_iterator()
iter = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
el = iter.get_next()

# create initialization ops
init_op = iter.make_initializer(data)

NR_EPOCHS = 1
with tf.Session() as sess:
    for e in range (NR_EPOCHS):
        print("\nepoch: ", e, "\n")
        sess.run(init_op)
        print("1  ", sess.run(el))
"""
