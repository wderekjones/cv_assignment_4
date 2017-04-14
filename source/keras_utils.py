import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import resize


def load_image_data_labels(path, sample_size):
    data = pd.read_csv(path)

    # if sample size is not provided then use all of the data
    if sample_size == None:
        sample_size = data.shape[0]
    data = data.sample(sample_size)

    labels = data["classid"]
    labels = labels.as_matrix()

    features = np.zeros([sample_size, 256, 256, 3])

    i = 0
    for filepath in data["filename"]:
        image_i = imread(filepath)
        image_i = resize(image_i, (256, 256, 3))
        features[i, :, :, :] = image_i

    for i in xrange(0, labels.shape[0]):
        labels[i] = labels[i] - 1

    return features, labels


def model0():
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=4, strides=1, dilation_rate=2, activation='tanh', input_shape=(4096, 1)))
    model.add(Conv1D(filters=1, kernel_size=2, activation='tanh'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(101, activation='softmax'))

    return model


def compile_model(model, learning_rate):
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
