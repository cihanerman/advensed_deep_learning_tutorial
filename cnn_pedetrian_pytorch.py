#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 14:31:19 2019

@author: cihanerman
"""

#%% import library

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#%% train preprocessing

def read_images(path, num_img):
    array = np.zeros([num_img, 64*32])
    i = 0
    
    for img in os.listdir(path):
        img_path = path + '//' + img
        img = Image.open(img_path, mode = 'r')
        data = np.asarray(img, dtype = 'uint8')
        data = data.flatten()
        array[i,:] = data
        i += 1
    
    return array

train_negative_path = r'LSIFIR/Classification/Train/neg'
num_train_negative_img = 43390
train_negative_array = read_images(train_negative_path,num_train_negative_img)

x_train_negative_tensor = torch.from_numpy(train_negative_array)
y_train_negative_tensor = torch.zeros(num_train_negative_img, dtype=torch.long)

train_positive_path = r'LSIFIR/Classification/Train/pos'
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path,num_train_positive_img)

x_train_positive_tensor = torch.from_numpy(train_positive_array)
y_train_positive_tensor = torch.ones(num_train_positive_img, dtype=torch.long)

#%% concat train

x_train = torch.cat((x_train_negative_tensor,x_train_positive_tensor), 0)
y_train = torch.cat((y_train_negative_tensor,y_train_positive_tensor), 0)

#%%

test_negative_path = r'LSIFIR/Classification/Test/neg'
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path,num_test_negative_img)

x_test_negative_tensor = torch.from_numpy(test_negative_array[:20855,:])
y_test_negative_tensor = torch.zeros(20855, dtype=torch.long)

test_positive_path = r'LSIFIR/Classification/Test/pos'
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path,num_test_positive_img)

x_test_positive_tensor = torch.from_numpy(test_positive_array)
y_test_positive_tensor = torch.ones(num_test_positive_img, dtype=torch.long)

#%% concat test

x_test = torch.cat((x_test_negative_tensor,x_test_positive_tensor), 0)
y_test = torch.cat((y_test_negative_tensor,y_test_positive_tensor), 0)

#%% visualize

plt.imshow(x_train[45001,:].reshape(64,32), cmap = 'gray')

#%% cnn model

num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.con1 = nn.Conv2d(1,10,5)
        self.pool = nn.MaxPool2d(2,2)
        self.con2 = nn.Conv2d(10,16,5)
        
        self.fc1 = nn.Linear(16*13*5,520)
        self.fc2 = nn.Linear(520,130)
        self.fc3 = nn.Linear(130,num_classes)
    
    def forward(self,x):
        x = self.pool(F.relu(self.con1(x)))
        x = self.pool(F.relu(self.con2(x)))
        x = x.view(-1,16*13*5) # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#%%

train = torch.utils.data.TensorDataset(x_train,y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size= batch_size, shuffle=True)

test = torch.utils.data.TensorDataset(x_test,y_test)
test_loader = torch.utils.data.DataLoader(test, batch_size= batch_size, shuffle=False)

#%%
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate, momentum=0.8)

#%% device confic EKSTRA DEFAULT CPU but GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ",device)
#%% train model

start = time.time()
train_acc = []
test_acc = []
lost_list =[]
use_gpu = False

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        
        inputs, labels = data
        inputs = inputs.view(batch_size, 1, 64, 32) # reshape
        inputs = inputs.float() # float
        
        # use gpu
        if use_gpu:
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
                
        # zero gradient
        optimizer.zero_grad()
        
        # forward
        outputs = model(inputs)
        
        # loss
        loss = criterion(outputs, labels)
        
        # back
        loss.backward()
        
        # update weights
        optimizer.step()
    
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels= data
            
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data,1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    acc1 = 100*correct/total
    print("accuracy test: ",acc1)
    test_acc.append(acc1)

    # train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_loader:
            images, labels= data
            
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data,1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    acc2 = 100*correct/total
    print("accuracy train: ",acc2)
    train_acc.append(acc2)


print("train is done.")

end = time.time()

print('process time: ', (end - start) / 60)
#%%