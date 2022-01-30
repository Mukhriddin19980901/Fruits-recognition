#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import glob
import os
import cv2
import cvlib as cv
from torch import from_numpy
import matplotlib.pyplot as plt
tf.config.list_physical_devices()


# In[19]:


def datasets(dire):
    x_train = []
    y_train = []
    b=[]
    a=np.zeros((5),dtype=np.uint8)
    label = ['freshapples','freshbanana','freshoranges','rottenapples','rottenbanana','rottenoranges']
    for i in range(len(label)):
        b=np.insert(a,i,1)
        path = dire+label[i]+'\\*'
        with open("size3.txt",'a+') as f:
                f.write(f"{path} |")
        for image in glob.glob(path):
            image = cv2.imread(image)
            image = cv2.resize(image,(128,128))
            x_train.append(image)
            y_train.append(b)
    return x_train,y_train
train_path = r'C:\\Users\\USER\\datasets2021\\meva\\dataset\\train\\'
test_path = r'C:\\Users\\USER\\datasets2021\\meva\\dataset\\test\\'
x_train=[]
y_train=[] 
x_train,y_train=datasets(train_path)


# In[20]:


x_train = np.array(x_train)
y_train=np.array(y_train)
print(x_train.shape,y_train.shape)


# In[21]:


x_test = []
y_test = []
x_test,y_test = datasets(test_path)
x_test = np.array(x_test)
y_test=np.array(y_test)
print(x_test.shape,y_test.shape)


# In[22]:


model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3) ,activation='relu' ,input_shape=(128,128,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.Dense(32,activation = 'relu'),
    keras.layers.Dense(16,activation = 'relu'),
    keras.layers.Dense(6,activation = 'sigmoid'),
])
tf.keras.layers.BatchNormalization(axis=-1,momentum = 0.99)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
with tf.device("/GPU:0"):
    model_gpu=model
    model_gpu.fit(x_train,y_train,epochs=7)
model_gpu.evaluate
model_gpu.summary


# In[23]:


model_gpu.evaluate(x_test,y_test)
model_gpu.summary()


# In[24]:


model_gpu.save('fruits1.model',save_format = 'h5')


# In[3]:


model = keras.models.load_model('fruits1.model')


# In[6]:


video = cv2.VideoCapture(r"videos/bananali.mp4")
label = ['olma','banan','apelsin','achigan olma','achigan banan','achigan apelsin']
while video.isOpened():
    _,image1=video.read()
    image=np.copy(image1)
    image=cv2.resize(image,(128,128))
    image=np.expand_dims(image,0)
    pred=model.predict(image)
    lab=label[np.argmax(pred)]
    if lab.startswith('achigan'):
        color=(255,0,0)
    else:
        color=(0,255,0)
    lab = f"{ lab }"
    cv2.putText(image1,lab,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,color,2)
    cv2.imshow("fruits",image1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video.release()
cv2.destroyAllWindows()


# In[ ]:




