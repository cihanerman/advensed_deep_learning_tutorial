#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 23:01:21 2019

@author: cihanerman
"""

# %% import library
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import json, codecs
import warnings
warnings.filterwarnings("ignore")

#%% load data and preproccessing
(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train), x_train.shape[1:][0]*x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test), x_test.shape[1:][0]*x_test.shape[1:][1]))

plt.imshow(x_train[4000].reshape(28,28))
plt.axis("off")
plt.show()

#%% Create Model

input_img = Input(shape = (x_train.shape[1],))
encoded1 = Dense(42, activation="relu")(input_img)
encoded2 = Dense(21, activation="relu")(encoded1)
decoded1 = Dense(42, activation="relu")(encoded2)
decoded2 = Dense(x_train.shape[1], activation="sigmoid")(decoded1)

model = Model(input_img, decoded2)
model.compile(optimizer="rmsprop", loss="binary_crossentropy")

#%% Model training

hist = model.fit(
            x_train,
            x_train,
            epochs=150,
            batch_size=300,
            shuffle=True,
            validation_data=(x_train, x_train)
        )

#%% save weight

model.save_weights("fashion_mnist_model_weight.h5", overwrite=True)

# %% save hist
with open("fashion_mnist_model_hist.json","w") as f:
    json.dump(hist.history,f)

# %% load history
with codecs.open("fashion_mnist_model_hist.json","r", encoding="utf-8")  as f:
    n = json.loads(f.read())
    
#%% evaluation
print(n.keys())

plt.plot(n["loss"],label = "Train loss")
plt.plot(n["val_loss"],label = "Val loss")

plt.legend()
plt.show()

#%% Predict
decoded_imgs = model.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.axis("off")

    # reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.axis("off")
plt.show()