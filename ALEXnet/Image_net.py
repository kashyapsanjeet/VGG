import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD



def Alex_net(no_of_output_unit):
    model = Sequential()

    #first convolutional layer
    model.add(Conv2D(96,kernel_size=(11,11),strides=(4,4),input_shape=(227,227,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization()) #Batch Normalization layer because no layer response normalization in keras
    
    #first maxpool_layer
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

    #second convolutional layer
    model.add(Conv2D(256,kernel_size=(5,5),strides=(1,1),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #second maxpool layer
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

    #third convolutional layer
    model.add(Conv2D(384,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #fourth convolutional layer
    model.add(Conv2D(384,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #fifth convolutional layer
    model.add(Conv2D(256,kernel_size=(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #third maxpool layer
    model.add(MaxPool2D(pool_size=(3,3),strides=(2,2)))

    model.add(Flatten())

    #first Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #second Dense layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    #third dense layer
    model.add(Dense(no_of_output_unit))
    model.add(Activation('softmax'))

    sgd=SGD(learning_rate=0.01,momentum=0.9)

    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

model = Alex_net(1000)
print(model.summary())
