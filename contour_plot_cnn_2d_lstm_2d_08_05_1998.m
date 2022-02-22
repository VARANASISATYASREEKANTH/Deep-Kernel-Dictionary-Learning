clc; 
clear all; 
close all;
AA_clstm_2=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\CNN_2D_LSTM_2D_out_matrix_08_05_1998.xlsx');
AA_clstm=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\CNN_LSTM_out_matrix_08_05_1998.xlsx');
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 30]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 30]);
time=0:13:1651;
height=41.9:0.3:80.1;

subplot(211)
contourf(time,height,((AA_clstm(1:128,2:129))./max(AA_clstm(1:128,2:129))),'linestyle','none','levelstep',0.1);
%hold on
%contourf(time,height,transpose((AA_clstm_2(1:128,1:128))./max(AA_clstm_2(1:128,1:128))),'linestyle','none','levelstep',0.1);
colormap('jet');
%caxis([0.7 1]);
colorbar;
ylim([60 80]);
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";


subplot(212)
%contourf(time,height,(AA_clstm(1:128,1:128))./max(AA_clstm(1:128,1:128)),'linestyle','none','levelstep',0.1);
%hold on
contourf(time,height,transpose((AA_clstm_2(1:128,1:128))./max(AA_clstm_2(1:128,1:128))),'linestyle','none','levelstep',0.1);
colormap('jet');
%caxis([0.7 1]);
colorbar;
ylim([60 80]);
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";
k=((cov(AA_clstm(1:128,2:129)))); 

eigen=(eig(k)/max(eig(k)));
s=svd(AA_clstm(1:128,2:129));

