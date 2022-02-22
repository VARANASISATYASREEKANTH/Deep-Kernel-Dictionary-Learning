# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:20:35 2021

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

import xlrd
import numpy as np
import matplotlib.pyplot as plt

import math
from random import *
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
sequence=np.zeros((184,1));

#-----data preprocessing-------
#df = pd.read_csv("G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/temperature_perturbations_hc_22_mar_2000_ver2.csv");
#df_t=df;
T_pert=np.zeros((184,40));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/temperature_perturbations_hc_22_mar_2000.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=sheet.cell_value(i,j);
#-------defining input sequence------
# define input sequence
sequence=T_pert[10][:];
#---------------------------------------------
#-----reshape input into [samples, timesteps, features]----
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))
#-----define model-------------
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
model.add(RepeatVector(n_in))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse');

# fit model
model.fit(sequence, sequence, epochs=300, verbose=0)
plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = model.predict(sequence, verbose=0)
print(yhat[0,:,0])