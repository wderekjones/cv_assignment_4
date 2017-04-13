import pandas as pd
import numpy as np
from skimage.transform import resize
from skimage.io import imread


from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, Dense, Flatten, Activation, Dropout
from keras.optimizers import Adam

def load_image_data_labels(path,sample_size):

    data = pd.read_csv(path)

    # if sample size is not provided then use all of the data
    if sample_size == None:
        sample_size = data.shape[0]
    data = data.sample(sample_size)


    labels = data["classid"]
    labels = labels.as_matrix()

    features =np.zeros([sample_size,256,256,3])

    i = 0
    for filepath in data["filename"]:
        image_i = imread(filepath)
        image_i = resize(image_i,(256,256,3))
        features[i,:,:,:] = image_i

    for i in xrange(0,labels.shape[0]):
        labels[i] = labels[i] - 1

    return features,labels

def model0():
    '''
    Constructs a convolutional model that takes as input a tensor of 3d images and maps to a probability distribution over class labels
    :return: an uncompiled model
    '''
    model = Sequential()
    model.add(Conv2D(filters=10,kernel_size=[3,3],data_format="channels_last",input_shape=(256,256,3),activation='relu'))
    model.add(Conv2D(filters=10,kernel_size=[3,3]))
    model.add(Flatten())
    model.add(Dense(101,activation='softmax'))

    return model

def model1(keep_prob):
    '''
    Constructs a fully connected model that takes as input the fc7 features of the alexNet model and maps to a probability distribution over class labels
    :return: an uncompiled model
    '''
    model = Sequential()
    model.add(Dense(2048,activation='tanh',input_shape=(4096,)))
    #model.add(Dropout(rate=keep_prob))
    model.add(Dense(1024,activation='tanh'))
    #model.add(Dropout(rate=keep_prob))
    model.add(Dense(512,activation='tanh'))
    #model.add(Dropout(rate=keep_prob))
    model.add(Dense(256,activation='tanh'))
    #model.add(Dropout(rate=keep_prob))
    model.add(Dense(128,activation='tanh'))
    model.add(Dense(101,activation='softmax'))

    return model

def compile_model(model,learning_rate):
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
