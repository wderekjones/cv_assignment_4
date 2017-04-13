from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from utils import *

codebook = make_codebook("data/metadata/train.csv",50,10)

surf_train_data,train_labels = get_codebook_features_labels("data/metadata/train.csv",codebook,500)
surf_test_data,test_labels = get_codebook_features_labels("data/metadata/test.csv",codebook,500)


logistic_clf = LogisticRegression()


scores = cross_val_score(logistic_clf,surf_train_data,train_labels,cv=5)

avg_score = np.mean(scores,axis=0)

print "random forest surf features accuracy (kfold cross-val train) "+str(avg_score)

logistic_clf.fit(surf_train_data,train_labels)

# evaluate classifier performance

logistic_train_preds = logistic_clf.predict(surf_train_data)
logistic_train_accuracy = accuracy_score(train_labels,logistic_train_preds)

logistic_test_preds = logistic_clf.predict(surf_test_data)
logistic_test_accuracy = accuracy_score(test_labels,logistic_test_preds)

print "random forest surf features accuracy (train-full): "+str(logistic_train_accuracy)
print "random forest surf features accuracy (test): "+str(logistic_test_accuracy)


confusion = confusion_matrix(test_labels,logistic_test_preds)

plot_confusion_matrix(confusion,np.arange(1,102),normalize=False,cmap=plt.cm.Purples)
plt.savefig("results/bow_logistic_confusion_results.jpg",dpi=600)
