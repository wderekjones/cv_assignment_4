import numpy as np
import matplotlib.pyplot as plt


from sklearn import svm
from skimage.io import imread,imshow
from utils import *
import h5py


train_data = make_dataset("surf")
