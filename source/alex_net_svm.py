from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import *

alex_net_train_data,train_labels = load_alex_net_image_features("data/metadata/train.csv")
alex_net_test_data, test_labels = load_alex_net_image_features("data/metadata/test.csv")



svm_clf = SVC()
svm_clf.fit(alex_net_train_data,train_labels)
# evaluate classifier performance

svm_train_preds = svm_clf.predict(alex_net_train_data)
svm_train_accuracy = accuracy_score(train_labels,svm_train_preds)
print train_labels.shape,svm_train_preds.shape

svm_test_preds = svm_clf.predict(alex_net_test_data)
svm_test_accuracy = accuracy_score(test_labels,svm_test_preds)
print test_labels.shape,svm_test_preds.shape
print "random forest surf features accuracy (train): "+str(svm_train_accuracy)
print "random forest surf features accuracy (test): "+str(svm_test_accuracy)