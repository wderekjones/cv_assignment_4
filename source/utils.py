import h5py
import pandas as pd
import numpy as np


def make_dataset(feature_type):
    train_data = pd.read_csv("data/metadata/train.csv")

    labels = train_data["classid"]

    min_feats = 99999999
    data_size = 0
    for filepath in train_data["filename"]:
        feats_i = load_surf_features(filepath)
        feats_i = np.transpose(feats_i)
        if feats_i.shape[0] < min_feats:
            min_feats = feats_i.shape[0]
        #print feats_i.shape[0]
        data_size+=1

    features = []

    if feature_type == "surf":
        i = 0
        features = np.zeros([data_size, (64 * min_feats)])

        for filepath in train_data["filename"]:
            feats_i = load_surf_features(filepath)
            feats_i = np.transpose(feats_i)
            feats_i = feats_i[0:min_feats,:]
            feats_i = np.ndarray.flatten(feats_i)
            features[i] = feats_i
            i+=1
    elif feature_type == "alex":
        i = 0
        features = np.zeros([data_size,4096])

        for filepath in train_data["filename"]:
            feats_i = load_alex_net_image_features(filepath)
            feats_i = np.ndarray.flatten(feats_i)
            features[i] = feats_i
            i+=1
    return features,labels

def load_surf_features(path):
    path = path+".h5"
    image_features = h5py.File(path)
    image_features = image_features.values()
    image_features[1] = np.array(image_features[1])
    image_features = image_features[1]

    return image_features

def load_alex_net_image_features(path):
    path = path +".h5"
    image_features = h5py.File(path)
    image_features = image_features.values()
    type(image_features)
    image_features[0] = np.array(image_features[0])
    image_features = image_features[0]

    return image_features
