# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:38:41 2021

@author: asdg
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:31:52 2021

@author: asdg
"""

import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
import keras
import sys
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

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import *
from pandas import ExcelWriter
from pandas import ExcelFile
from math import log 
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
params = {'backend': 'ps',
          'axes.labelsize': 20,
          'legend.fontsize': 16,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True};

T_pert=np.zeros((184,40));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/temperature_perturbations_hc_22_mar_2000.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=sheet.cell_value(i,j);

#--------------reshaping the data-------------






cnn = Sequential()
cnn.add(Conv2D(1, (2,2), activation='relu', padding='same', input_shape=(10,10,1)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())