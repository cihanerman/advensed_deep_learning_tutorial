#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:41:04 2019

@author: cihanerman
"""
#%% libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
#%% import data
train = pd.read_csv('mnist-in-csv/mnist_train.csv', sep=',')
#train = train.values
test = pd.read_csv('mnist-in-csv/mnist_test.csv', sep=',')
#test = test.values

img_size = 28

Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)

Y_test = test['label']
X_test = test.drop(labels=['label'], axis=1)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.values.reshape(-1, img_size, img_size, 1)
X_test = X_test.values.reshape([-1, img_size, img_size, 1])

Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)
#%% img plot
v = X_train.reshape(X_train.shape[0],img_size,img_size)
plt.imshow(v[13, :, :], cmap='gray')
plt.axis('off')
plt.show()

#%% model
number_of_class = 10
input_shape = (img_size, img_size, 1)

model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_class)) #output layer
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#%%
batch_size = 5000
hist = model.fit(X_train,Y_train, validation_data=(X_test, Y_test), epochs=25, batch_size=batch_size)
#%% model save
model.save_weights('cnn_mnis_model_weights.h5')
#%% model evalation
print(hist.history.keys())
plt.plot(hist.history['loss'], label='Trainin loss')
plt.plot(hist.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history['acc'], label='Trainin acc')
plt.plot(hist.history['val_acc'], label='Validation acc')
plt.legend()
plt.show()
#%% save history
import json
with open('cnn_mnist_hist.json', 'w') as f:
    json.dump(hist.history, f)


