from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import *

codebook = make_codebook("data/metadata/train.csv",50,20)

surf_train_data,train_labels = get_codebook_features_labels("data/metadata/train.csv",codebook)
surf_test_data,test_labels = get_codebook_features_labels("data/metadata/test.csv",codebook)



rForest_clf = RandomForestClassifier(n_estimators = 100)
rForest_clf.fit(surf_train_data,train_labels)

# evaluate classifier performance

forest_train_preds = rForest_clf.predict(surf_train_data)
forest_train_accuracy = accuracy_score(train_labels,forest_train_preds)

forest_test_preds = rForest_clf.predict(surf_test_data)
forest_test_accuracy = accuracy_score(test_labels,forest_test_preds)

print "random forest surf features accuracy (train): "+str(forest_train_accuracy)
print "random forest surf features accuracy (test): "+str(forest_test_accuracy)

