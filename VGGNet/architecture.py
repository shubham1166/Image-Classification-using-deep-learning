'''
Created on Jun 13, 2019
This code has the model that we are using in case of classification in cifar 10
@author: T01130
'''
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Flatten,BatchNormalization
def VGG_seq(imheight=28,imwidth=28,imdepth=1,classes=10,path2load=None):
    VGG=Sequential()
    VGG.add(Conv2D(64,kernel_size=(3,3),padding='same',input_shape=(imheight,imwidth,imdepth)))
    VGG.add(BatchNormalization())
    VGG.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
    VGG.add(BatchNormalization())
    VGG.add(MaxPool2D(pool_size=(2,2)))
    VGG.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
    VGG.add(BatchNormalization())
    VGG.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
    VGG.add(BatchNormalization())
    VGG.add(MaxPool2D(pool_size=(2,2)))
    VGG.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
    VGG.add(BatchNormalization())
    VGG.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
    VGG.add(BatchNormalization())
    VGG.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same'))
    VGG.add(MaxPool2D(pool_size=(2,2)))
    VGG.add(Flatten())
    VGG.add(Dense(128,activation='relu'))
    VGG.add(Dropout(0.5))
    VGG.add(Dense(classes,activation='softmax'))
    #Loading weiths if given
    if path2load is not None:
        VGG.load_weights(path2load)
    return VGG