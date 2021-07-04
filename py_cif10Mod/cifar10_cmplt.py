# -*- coding: utf-8 -*-


from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv

(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()

X_train,X_test=X_train/255,X_test/255
Classes=['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(16):
  plt.subplot(4,4,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(X_train[i],cmap=plt.cm.binary)
  plt.xlabel(Classes[y_train[i][0]])

model =models.Sequential()
model.add(layers.Conv2D(filters=45,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(filters=60,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(70,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

model.fit(X_train,y_train,epochs=12)

L,A=model.evaluate(X_test,y_test)
print(f'Loss: {L} , Accuracy: {A}')

model.summary()

model.save('Model_cifar10')