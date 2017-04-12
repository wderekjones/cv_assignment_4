import h5py
import pandas as pd
import numpy as np


from sklearn.cluster import KMeans


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
        data_size+=1

    features = []

    if feature_type == "surf":
        i = 0

        codebook_feats = []
        for filepath in train_data["filename"]:
            feats_i = load_surf_features(filepath)
            feats_i = np.transpose(feats_i)

            feats_i = feats_i[0:min_feats,:]
            for feat in feats_i:
                codebook_feats.append(feat)

            i+=1
        codebook_feats = np.asarray(codebook_feats)
        codebook = make_codebook(50,codebook_feats)

        del codebook_feats

        # for each of the training examples, using the codebook, make a histogram over the set of all of the respective surf features. normalize this and consider as the corresponding feature vector

        features = np.zeros([train_data.shape[0],50])

        i = 0

        for filepath in train_data["filename"]:
            feats_i = load_surf_features(filepath)
            feats_i = np.transpose(feats_i)

            hgram_i = np.zeros([1,50])

            denom = feats_i.shape[0]

            for feat in feats_i:
                f_i = codebook.predict([feat])
                hgram_i[0,f_i] +=1

            hgram_i = np.divide(hgram_i,float(denom))

            features[i] = hgram_i

        features = np.asarray(features)
        print  features.shape
    elif feature_type == "alex":
        i = 0
        features = np.zeros([data_size,4096]) # try making this a smaller number in order to imporve speed, try dimensionality reduction?

        for filepath in train_data["filename"]:
            feats_i = load_alex_net_image_features(filepath)
            feats_i = np.ndarray.flatten(feats_i)
            features[i] = feats_i
            i+=1
    return features,labels

def make_codebook(num_words,X):
    cluster = KMeans(n_clusters=num_words)
    cluster.fit(X)
    return cluster

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
