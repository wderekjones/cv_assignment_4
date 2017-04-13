from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from utils import *

alex_net_train_data, train_labels = get_alex_feats_labels("data/metadata/train.csv",500)
alex_net_test_data, test_labels = get_alex_feats_labels("data/metadata/test.csv",500)


rForest_clf = RandomForestClassifier(n_estimators=500)

scores = cross_val_score(rForest_clf,alex_net_train_data,train_labels,cv=5)

avg_score = np.mean(scores,axis=0)

print "random forest surf features accuracy (kfold cross-val train) "+str(avg_score)


rForest_clf.fit(alex_net_train_data,train_labels)
# evaluate classifier performance

svm_train_preds = rForest_clf.predict(alex_net_train_data)
svm_train_accuracy = accuracy_score(train_labels,svm_train_preds)
print train_labels.shape,svm_train_preds.shape

svm_test_preds = rForest_clf.predict(alex_net_test_data)
svm_test_accuracy = accuracy_score(test_labels,svm_test_preds)

print "random forest surf features accuracy (train): "+str(svm_train_accuracy)
print "random forest surf features accuracy (test): "+str(svm_test_accuracy)

confusion = confusion_matrix(test_labels,svm_test_preds)

plot_confusion_matrix(confusion,np.arange(1,102),normalize=False,cmap=plt.cm.Purples)
plt.savefig("results/alex_net_random_forest_confusion_results.jpg",dpi=600)
