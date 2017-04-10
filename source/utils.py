import h5py
import pandas as pd
import numpy as np


def make_dataset(feature_type):
    train_data = pd.read_csv("data/metadata/train.csv")
    features = np.array([])
    print features.shape


    if feature_type == "surf":
        for filepath in train_data["filename"]:
            feats_i = load_surf_features(filepath)
            feats_i = np.transpose(feats_i)
            print feats_i.shape
            #features = np.concatenate((features,feats_i),axis=0)
    elif feature_type == "alex":
        for filepath in train_data["filename"]:
            feats_i = load_alex_net_image_features(filepath)
            print feats_i.shape
            #features = np.concatenate((features,feats_i),axis=0)
    return features

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
    image_features[0] = np.array(image_features[0])
    image_features = image_features[0]

    return image_features
