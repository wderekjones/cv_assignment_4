import pandas as pd
import numpy as np
from skimage.transform import resize
from skimage.io import imread


from keras.models import Sequential
from keras.layers import Conv2D, Conv3D, Dense, Flatten, Activation
from keras.optimizers import Adam

def load_image_data_labels(path,sample_size):

    data = pd.read_csv(path)

    data = data.sample(sample_size)

    labels = data["classid"]

    features =np.zeros([sample_size,256,256,3])

    i = 0
    for filepath in data["filename"]:
        image_i = imread(filepath)
        image_i = resize(image_i,(256,256,3))
        features[i,:,:,:] = image_i

    return features,labels

def deep_model():
    model = Sequential()
    model.add(Conv3D(filters=10,kernel_size=[3,3,3],strides=1,dilation_rate=1,input_shape=(None,256,256,3)))
    model.add(Conv3D(filters=10,kernel_size=[3,3,3],dilation_rate=1))
    model.add(Flatten())
    #model.add(Dense(activation='relu'))
    model.add(Dense(101,activation='softmax'))

    return model

def compile_model(model):
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
