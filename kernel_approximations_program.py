# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:39:04 2021

@author: asdg
"""

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import time
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import keras
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import *
from math import log 
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
y=np.zeros(128);
y[100:120]=np.ones(20);
X=np.zeros((128,128));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/Aura_MLS/AURA_DATA_April_2014/y_DL_21_04_2014_64_samples.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        X[i][j]=sheet.cell_value(i,j)

rbf_feature = RBFSampler(gamma=1, n_components=64,  random_state=1);
X_features = rbf_feature.fit_transform(X);
clf = SGDClassifier(max_iter=5, tol=1e-3);
clf.fit(X_features, y);
SGDClassifier(max_iter=5);
clf.score(X_features, y);