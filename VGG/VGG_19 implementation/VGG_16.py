import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


#input shape of image = (224 * 224 * 3)
#first convolution kernel size size = 3*3
#second convolutional filter size = 1*1
#stride = 1
#padding = same
#maxpooling = 2*2 and stride = 2

#architecture
#conv3-64 -> maxpool -> conv3-128 -> maxpool -> conv3-256 -> conv3-256 -> maxpool -> conv3-512 -> conv3-512 -> maxpool -> conv3-512 -> conv3-512 -> maxpool -> FC-4096 -> FC-4096 -> FC-1000 -> Softmax
def Vgg_11(num_of_classes):


    model = Sequential()
    #first convolution
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',input_shape = (224,224,3) ))
    model.add(Activation('relu'))

    #second convolution
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same' ))
    model.add(Activation('relu'))

    #first pooling layer
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))

    #third Convolution
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))
    
    #fourth convolution
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #second pooling layer
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))

    #fifth convolution
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #sixth convolutional layer
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #seventh convolutional layer
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #third Pooling Layer
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))

    #eigth convolution
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #ninth convolution
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #tenth convolution
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #fourth pooling layer
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))

    #eleventh convolution
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #twelth convolution
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #Thirteenth convolution
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same'))
    model.add(Activation('relu'))

    #fifth pooling layer
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same'))

    model.add(Flatten())

    #Fully connected layers
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(num_of_classes))  #no of output nodes = no_of_classes
    model.add(Activation('softmax'))

    sgd = SGD(learning_rate=0.01)

    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model

model = Vgg_11(4)
print(model.summary())