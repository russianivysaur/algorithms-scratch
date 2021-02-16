from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hdf5storage
import numpy
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
if not os.path.isdir("tumorpics"):
        os.mkdir("tumorpics")
        os.mkdir("tumorpics/masks")
        os.mkdir("tumorpics/mri")
images = []
masks = []
for i in range(1,3064):
            st = "tumors/tumors/"+str(i)+".mat"
            mat = hdf5storage.loadmat(st)
            mat_file = mat['cjdata'][0]
            image = cv2.resize(mat_file[2], dsize=(256,256),
                       interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mat_file[4].astype('float32'), dsize=(256,256),
                      interpolation=cv2.INTER_CUBIC)
            #cv2.imwrite("tumorpics/mri/"+str(i)+".jpg",image)
            #cv2.imwrite("tumorpics/masks/"+str(i)+".jpg",mask)
            images.append(image)
            masks.append(mask)


def model():
       model = Sequential()
       model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(256,256,1)))
       model.add(MaxPooling2D((2,2),strides=(2,2)))
       model.add(BatchNormalization(momentum=0.6))
       model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
       model.add(MaxPooling2D((2,2),strides=(2,2)))
       model.add(BatchNormalization(momentum=0.6))
       model.add(Dropout(0.4))
       model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
       model.add(MaxPooling2D((2,2),strides=(2,2)))
       model.add(BatchNormalization())
       model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
       model.add(MaxPooling2D((2,2),strides=(2,2)))
       model.add(BatchNormalization(momentum=0.6))
       model.add(Dropout(0.3))
       model.add(Conv2D(256,(1,1),activation='relu',padding='same'))
       model.add(Conv2DTranspose(128,(3,3),strides=(2,2),activation='relu',padding='same'))
       model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
       model.add(Dropout(0.3))
       model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
       model.add(Conv2DTranspose(64,(3,3),activation='relu',padding='same',strides=(2,2)))
       model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
       model.add(Dropout(0.3))
       model.add(Conv2DTranspose(32,(3,3),activation='relu',padding='same',strides=(2,2)))
       model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
       model.add(Dropout(0.3))
       model.add(Conv2DTranspose(32,(3,3),activation='relu',padding='same',strides=(2,2)))
       model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
       model.add(Dropout(0.3))
       model.add(Conv2D(1,(1,1),activation='relu'))
       return model


from keras.losses import binary_crossentropy
from keras import backend as K
import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
images = numpy.expand_dims(images,axis=-1)
images = numpy.array(images)
masks = numpy.expand_dims(masks,axis=-1)
masks = numpy.array(masks)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(images,masks,test_size=0.3)
from tensorflow.keras.models import load_model
model = model()
model.compile(loss=bce_dice_loss,optimizer='adam',metrics=['accuracy'])
model.fit(xtrain,ytrain,validation_data=(xtest,ytest),verbose=1,epochs=100)
