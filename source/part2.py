import numpy as np
import matplotlib.pyplot as plt


from sklearn import svm
from skimage.io import imread,imshow
from utils import *
import h5py


surf_train_data = make_dataset("surf")
alex_train_data = make_dataset("alex")
print surf_train_data
print alex_train_data