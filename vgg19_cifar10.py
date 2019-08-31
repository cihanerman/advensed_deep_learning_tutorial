#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:51:12 2019

@author: cihanerman
"""

# %% import library
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
import cv2
import numpy as np
# %% preprocessing 

batch_size = 1000
num_classes = 10
epochs = 5

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#%% visualize
plt.imshow(x_train[6].astype(np.uint8))
plt.axis('off')
plt.show()

#%% reshape
def resize_img(img, width, height, color_scale):
    number_of_images = img.shape[0]
    new_array = np.zeros((number_of_images, width, height, color_scale))
    for i in range(number_of_images):
        new_array[i] = cv2.resize(img[i,:,:,:], (width, height))
    return new_array

x_train = resize_img(x_train, 48, 48, 3)
x_test = resize_img(x_test, 48, 48, 3)

input_shape = x_train.shape[1:]

plt.figure()
plt.imshow(x_train[6].astype(np.uint8))
plt.axis('off')
plt.show()
# %% model

vgg = VGG19(include_top = False, weights = 'imagenet', input_shape = input_shape)
print(vgg.summary())

vgg_layers = vgg.layers
model = Sequential()

for i in range(len(vgg_layers) - 1):
    model.add(vgg_layers[i])

for layer in model.layers:
    layer.trainable = False

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())
model.compile(
            loss ='categorical_crossentropy',
            optimizer = 'rmsprop',
            metrics = ['accuracy']
        )

#%% model train

hist = model.fit(
            x_train,
            y_train,
            validation_split = 0.2,
            epochs = epochs,
            batch_size = batch_size
        )

#%% model and history save

model.save_weights('vgg19_cifar10.h5')

import json, codecs
with open("vgg19_cifar10.json","w") as f:
    json.dump(hist.history,f)

#%% load history and visualize

with codecs.open("vgg19_cifar10.json","r", encoding = "utf-8") as f:
    n = json.loads(f.read())

plt.plot(n["loss"],label = "training loss")
plt.plot(n["val_loss"],label = "validation loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(n["acc"],label = "training acc")
plt.plot(n["val_acc"],label = "validation acc")
plt.legend()
plt.show()