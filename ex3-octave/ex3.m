%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%  







%  2022年6月11日15:09:54：
%  多分类逻辑回归 + 神经网络 实现多分类任务
%  任务是识别手写的图片是0-9中某一个数字














%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size = 400; % 20x20 Input Images of Digits
num_labels = 10; % 10 labels, from 1 to 10
% (note that we have mapped "0" to label 10)












%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat');       % training data stored in arrays X, y
m = size(X, 1);             % 样本数目，在这里为5000

% X是5000*400，在这里是5000个样本，每个样本是一个20*20像素的灰度图像
% y是5000*1; y的取值有10种，分别是0-9，表示图片识别结果，
% 在本实验中，为了方便表示，y中1-9表示1-9,10表示识别结果为0
size(X)                     
size(y)                     


% Randomly select 100 data points to display
% 随机选取100个样本索引index行，用来可视化
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

% 可视化一下随机抽取的那100个图片样本
displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;












%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. Your task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
% 下面用一个简单的测试用例测试一下lrCostFunction函数，该函数中需要由你实现向量化计算梯度和损失
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];                           % 初始化θ，4*1
X_t = [ones(5, 1), reshape(1:15, 5, 3) / 10];       % 初始化X，5*4
y_t = ([1; 0; 1; 0; 1] >= 0.5);                     % 初始化y，5*1
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
