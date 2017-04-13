import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def get_codebook_features_labels(path,codebook,sample_size):
    train_data = pd.read_csv(path)

    if sample_size == None:
        sample_size = train_data.shape[0]

    # select a subset of the data of size sample_size to make the computation more efficient
    train_data = train_data.sample(sample_size)

    labels = train_data["classid"]
    labels = labels.as_matrix()

    features = np.zeros([train_data.shape[0],codebook.cluster_centers_.shape[0]])

    i = 0
    for filepath in train_data["filename"]:
        feats_i = load_surf_features(filepath)
        feats_i = np.transpose(feats_i)

        hgram_i = np.zeros([1,codebook.cluster_centers_.shape[0]])

        for feat in feats_i:
            f_i = codebook.predict([feat])
            hgram_i[0,f_i] +=1

        hgram_i = np.divide(hgram_i,float(feats_i.shape[0]))

        features[i] = hgram_i   #try features[i,:] = hgram_i
        i += 1

    features = np.asarray(features)

    for i in xrange(0,labels.shape[0]):
        labels[i] = labels[i] - 1

    return features,labels


def get_alex_feats_labels(path,sample_size):
    train_data = pd.read_csv(path)

    if sample_size == None:
        sample_size = train_data.shape[0]
    # in order to decrease computation burden, select a random sample of the rows equivalent of size sample_size
    train_data = train_data.sample(sample_size)

    labels = train_data["classid"]
    labels = labels.as_matrix()

    i = 0
    features = np.zeros([sample_size,4096])

    for filepath in train_data["filename"]:
        feats_i = load_alex_net_image_features(filepath)
        feats_i = np.ndarray.flatten(feats_i)
        features[i] = feats_i
        i+=1

    for i in xrange(0,labels.shape[0]):
        labels[i] = labels[i] - 1

    return features,labels


def make_codebook(path,size,num_words):
    data = pd.read_csv(path)

    codebook_feats = []
    for filepath in data["filename"]:
        feats_i = load_surf_features(filepath)
        feats_i = np.transpose(feats_i)

        if feats_i.shape[0] >= num_words: # throw out samples with smaller number of features than the min number of features we would like to extract
            rand_indices = np.unique(np.random.choice(feats_i.shape[0],num_words,replace=False))
            for index in rand_indices:
                codebook_feats.append(feats_i[index])

    codebook_feats = np.asarray(codebook_feats)
    cluster = KMeans(n_clusters=size,max_iter=300)
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
    plt.xticks(tick_marks, classes, rotation=45,fontsize=2)
    plt.yticks(tick_marks, classes,fontsize=2)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #thresh = cm.max() / 2.

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

