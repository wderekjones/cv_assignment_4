import h5py
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_codebook_features_labels(path,codebook):
    train_data = pd.read_csv(path)
    labels = train_data["classid"]

    features = np.zeros([train_data.shape[0],50])

    i = 0
    for filepath in train_data["filename"]:
        feats_i = load_surf_features(filepath)
        feats_i = np.transpose(feats_i)

        hgram_i = np.zeros([1,50])

        for feat in feats_i:
            f_i = codebook.predict([feat])
            hgram_i[0,f_i] +=1

        hgram_i = np.divide(hgram_i,float(feats_i.shape[0]))

        features[i] = hgram_i   #try features[i,:] = hgram_i
        i += 1

    features = np.asarray(features)

    return features,labels

def get_alex_feats_labels(path):
    train_data = pd.read_csv(path)
    labels = train_data["classid"]

    i = 0
    features = np.zeros([train_data.shape[0],4096]) # try making this a smaller number in order to imporve speed, try dimensionality reduction?

    for filepath in train_data["filename"]:
        feats_i = load_alex_net_image_features(filepath)
        feats_i = np.ndarray.flatten(feats_i)
        features[i] = feats_i
        i+=1
    return features,labels


def make_codebook(path,size,num_words):
    data = pd.read_csv(path)

    codebook_feats = []
    for filepath in data["filename"]:
        feats_i = load_surf_features(filepath)
        feats_i = np.transpose(feats_i)

        #rand_indices = np.random.randint(0, feats_i.shape[0], size=num_words)   #make sure this doesn't generate duplicates
        if feats_i.shape[0] >= num_words: # throw out samples with smaller number of features than the number of features we would like to extract
            rand_indices = np.random.choice(feats_i.shape[0],num_words,replace=False)
            for index in rand_indices:
                codebook_feats.append(feats_i[index])

    codebook_feats = np.asarray(codebook_feats)
    cluster = KMeans(n_clusters=size,max_iter=1000)
    cluster.fit(codebook_feats)

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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

