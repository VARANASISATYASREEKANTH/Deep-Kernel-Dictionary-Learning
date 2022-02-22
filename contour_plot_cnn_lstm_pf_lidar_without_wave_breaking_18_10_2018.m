clc; 
clear all; 
close all;
AA_clstm=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\CNN_LSTM_out_matrix_pfl_18_10_2018_512_points.xlsx');
k=AA_clstm./max(AA_clstm);
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [30 10]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 30 10]);
time=0:13:1651;
height=80.013:0.06:95.313;

subplot(211)
contourf(time,height,k(1:256,1:128),'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([0.7 1]);
colorbar;
ylim([80 95]);
xticks([0:60:360])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";