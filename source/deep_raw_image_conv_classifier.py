from keras_utils import *
import matplotlib.pyplot as plt

data,labels = load_image_data_labels("data/metadata/train.csv",4)

model = deep_model()

compile_model(model)

model.fit(data,labels,batch_size=5,nb_epoch=100,shuffle=True,validation_split=0.2)