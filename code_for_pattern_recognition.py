# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:49:42 2021

@author: asdg
"""
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
        
        
        
        
        
