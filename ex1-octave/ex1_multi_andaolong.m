%% 机器学习在线课程实验1：线性回归
% 第二部分：可选实验：
%           1.梯度下降法实现 多变量线性回归
%           2.正规方程法实现 多变量线性回归



%  本方法用到的函数：
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%   本函数用到的变量及介绍：
%       单变量回归中:h(x) = θ_0 + θ_1 * x
%       m 一般用来表示样本个数
%       n 一般用来表示feature属性个数
%       X 是feature属性值，是城市人口；在函数中有时会加一列截距项；m行n列或n+1列
%       y 是Label标签，是在该城市一辆小吃车的利润；m行1列
%       theta 也就是θ，我们最终要求得的参数向量；2行一列，因为单变量回归只有两个参数θ0和θ1
%       iterations 梯度下降法的迭代次数
%       alpha   梯度下降法的学习率




%% Initialization
clear ; close all; clc














%% ================ Part 1: Feature Normalization ================
%% =================  特征归一化==================================

%  1.读取数据
fprintf('Loading data ...\n');

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);


% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

% X是归一化后的源数据,mu是每一个feature的平均值,sigma是每一个feature的标准差
[X mu sigma] = featureNormalize(X);


fprintf('输出一些归一化之后的X \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Add intercept term to X, 给X添加截距项
X = [ones(m, 1) X]

% ============================================================













%% ================ Part 2: Gradient Descent ================
%% ================ Part 2: 梯度下降法实现多变量线性回归 ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.01;
num_iters = 300;

% Init Theta and Run Gradient Descent 
% 这里我多设置了几个学习率，比较一下收敛的快慢
theta = zeros(3, 1);
theta2 = zeros(3, 1);
theta3 = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
[theta2, J_history2] = gradientDescentMulti(X, y, theta2, 0.03, num_iters);
[theta3, J_history3] = gradientDescentMulti(X, y, theta3, 0.1, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 1);
hold on;
plot(1:numel(J_history2), J_history2, '-r', 'LineWidth', 1);
hold on;
plot(1:numel(J_history3), J_history3, '-k', 'LineWidth', 1);

xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this
X_estimate = [1650, 3];
% 这里一定要注意，前面X进行了特征归一化所以这里也要进行特征归一化
X_estimate = (X_estimate - mu) ./ sigma;   
X_estimate = [ones(1,1) X_estimate]
price = X_estimate * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;


















%% ================ Part 3: Normal Equations ================
%% ================ 正规方程法 实现多变量线性回归 ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
% 这是关键，调用normalEqn()方法
theta = normalEqn(X, y);    

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% 在这里，theta已经求出来了，是3行1列
price = 0; % You should change this
X_estimate = [1, 1650 , 3];
price = X_estimate * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

