import numpy as np
import matplotlib.pyplot as plt


from sklearn import svm
from skimage.io import imread,imshow
from utils import *
import h5py


surf_train_data,labels1 = make_dataset("surf")
alex_train_data, labels2 = make_dataset("alex")

clf = svm.SVC()
clf.fit(alex_train_data,labels1)

# evaluate classifier performance
num_correct = 0

for i in xrange(0,alex_train_data.shape[0]):
    pred = clf.predict(alex_train_data[i,:])
    if pred == labels1[i]:
        num_correct +=1

print "surf features accuracy: "+str(num_correct/float(surf_train_data.shape[0]))


