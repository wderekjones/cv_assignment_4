from sklearn import svm
from sklearn.metrics import accuracy_score
from utils import *


surf_train_data,labels1 = make_dataset("surf")

clf = svm.SVC()
clf.fit(surf_train_data,labels1)

# evaluate classifier performance
num_correct = 0


preds = clf.predict(surf_train_data)

surf_accuracy = accuracy_score(labels1,preds)

print "surf features accuracy: "+str(surf_accuracy)