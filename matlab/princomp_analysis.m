clear; close all; clc;

%% An attempt of applying PCA 
data = csvread('../data/preprocessed.csv',1,1);
train_X = data(1:99,2:end);
train_y = data(1:99,1); 
test_X = data(100:end,2:end); 

[coeff,score,latent,tsquared,explained,mu] = pca([train_X;test_X]);
no_comp = 3;
sum(explained(1:no_comp))/100