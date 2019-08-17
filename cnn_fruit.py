#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:43:15 2019

@author: cihanerman
"""
#%% libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob
#%%
train_path = 'fruits/fruits-360/Training/'
test_path = 'fruits/fruits-360/Test/'

img = load_img(train_path + 'Apple Braeburn/0_100.jpg')
plt.imshow(img)
plt.axis('off')
plt.show()

x = img_to_array(img)
print(x.shape)

class_names = glob(train_path + '/*')
number_of_class = len(class_names)
#%% cnn
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=x.shape))
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

#%% trainin - test - data generation
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                             shear_range=0.3,
                             horizontal_flip=True,
                             zoom_range=0.3)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generation = train_datagen.flow_from_directory(train_path,
                                                     target_size=x.shape[:2],
                                                     batch_size=batch_size,
                                                     color_mode='rgb',
                                                     class_mode='categorical')

test_generation = test_datagen.flow_from_directory(test_path,
                                                   target_size=x.shape[:2],
                                                   batch_size=batch_size,
                                                   color_mode='rgb',
                                                   class_mode='categorical')

hist = model.fit_generator(
            generator=train_generation,
            steps_per_epoch=1600 // batch_size,
            epochs=100,
            validation_data=test_generation,
            validation_steps=800 // batch_size
        )

#%% model save
model.save_weights('cnn_fruits_model_weights.h5')
#%% model evalation
print(hist.history.keys())
plt.plot(hist.history['loss'], label='Trainin loss')
plt.plot(hist.history['val_loss'], label='Validation loss')
plt.plot(hist.history['acc'], label='Trainin acc')
plt.plot(hist.history['val_acc'], label='Validation acc')
plt.legend()
plt.show()

#%% save history
import json
with open('cnn_fruit_hist.json', 'w') as f:
    json.dump(hist.history, f)
    
#%% load history
import codecs
with codecs.open('cnn_fruit_hist.json', 'r', encoding='utf-8') as f:
    h = json.loads(f.read())
    
plt.plot(h['loss'], label='Trainin loss')
plt.plot(h['val_loss'], label='Validation loss')
plt.plot(h['acc'], label='Trainin acc')
plt.plot(h['val_acc'], label='Validation acc')
plt.legend()
plt.show()

#%%
