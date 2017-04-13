from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

from keras_utils import *
from utils import *

train_data, train_labels = get_alex_feats_labels("data/metadata/train.csv",None)
test_data, test_labels = get_alex_feats_labels("data/metadata/test.csv",None)


train_labels = to_categorical(train_labels, 101)

test_labels = to_categorical(test_labels, 101)

model = model1(0.5)

compile_model(model,0.001)

model.fit(train_data, train_labels, batch_size=64, nb_epoch=100, shuffle=True, validation_split=0.2)

print test_labels.shape

deep_preds = model.predict(test_data)
deep_preds = np.argmax(deep_preds, axis=1)
test_labels = np.argmax(test_labels, axis=1)

deep_accuracy = accuracy_score(test_labels, deep_preds)

deep_confusion = confusion_matrix(test_labels, deep_preds, labels=np.arange(0, 101))

plot_confusion_matrix(deep_confusion,np.arange(0,101),normalize=False,cmap=plt.cm.Purples)

plt.show()
