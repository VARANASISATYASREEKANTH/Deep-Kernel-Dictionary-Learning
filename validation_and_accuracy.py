# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:40:40 2021

@author: asdg
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from pylab import rcParams
import tensorflow as tf
import keras
from keras_layer_normalization import LayerNormalization
from keras.layers import LayerNormalization
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed
from keras import optimizers, Sequential

from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score 
from keras.layers import LayerNormalization
model1=Sequential()
#model1.add(Embedding(3,5))
model1.add(LSTM(32));
model1.add(Dense(1,activation='sigmoid'));
model1.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['acc']);
history=model1.fit(x_train, y_train, batch_size=200, epochs=10, validation_split=0.2);





