from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from utils import *

codebook = make_codebook("data/metadata/train.csv",50,10)

surf_train_data,train_labels = get_codebook_features_labels("data/metadata/train.csv",codebook,500)
surf_test_data,test_labels = get_codebook_features_labels("data/metadata/test.csv",codebook,500)


rForest_clf = RandomForestClassifier(n_estimators = 100,criterion="entropy")


scores = cross_val_score(rForest_clf,surf_train_data,train_labels,cv=5)

avg_score = np.mean(scores,axis=0)

print "random forest surf features accuracy (kfold cross-val train) "+str(avg_score)

rForest_clf.fit(surf_train_data,train_labels)

# evaluate classifier performance

forest_train_preds = rForest_clf.predict(surf_train_data)
forest_train_accuracy = accuracy_score(train_labels,forest_train_preds)

forest_test_preds = rForest_clf.predict(surf_test_data)
forest_test_accuracy = accuracy_score(test_labels,forest_test_preds)

print "random forest surf features accuracy (train-full): "+str(forest_train_accuracy)
print "random forest surf features accuracy (test): "+str(forest_test_accuracy)


confusion = confusion_matrix(test_labels,forest_test_preds)

plot_confusion_matrix(confusion,np.arange(1,102),normalize=False,cmap=plt.cm.Purples)
plt.savefig("results/bow_random_forest_confusion_results.jpg",dpi=600)
