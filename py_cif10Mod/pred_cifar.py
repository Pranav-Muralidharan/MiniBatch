from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np 
import cv2 as cv

model=models.load_model('Model_cifar10')
#img=cv.imread('......')
#img=cv.cvtColor(img,cv.COLOR_BGR2RGB)

#plt.imshow(img,Cmap=plt.cm.binary)
#P=model.predict(np.array([img])/255)
#print(f'prediction: {Classes[np.argmax(P)]}')