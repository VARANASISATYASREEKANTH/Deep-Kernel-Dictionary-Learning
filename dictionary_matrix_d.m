clc;
clear all;
[d,txt,raw] = xlsread('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/D_32x32.xlsx');
[in,txt,raw] = xlsread('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data_pert_21_04_2014.xlsx');
RESULT=d(1:128,1:128).*in(1:128,1:128);