%% 机器学习在线课程实验1：线性回归



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










%% ===================== 1.读取数据 =========================
fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1);     % X是城市的人口,是feature
y = data(:, 2);     % y是在该城市一辆车的利润,是Label
m = length(y);      % m是训练样本的数目
          
plotData(X, y)      % 绘制一下原始数据，在这里调用自己写的plotData函数

fprintf('Program paused. Press enter to continue.\n\n\n');
pause;
%% ===============================================














%% =======================2.计算损失和进行梯度下降========================
X = [ones(m, 1), data(:, 1)];       % X前面加一列截距项
                                    
theta = zeros(2, 1);                % initialize fitting parameters

iterations = 1500;                  % 设置一些初始参数
alpha = 0.01;

theta = gradientDescent(X, y, theta, alpha, iterations);    % 调用函数进行梯度下降求出theta


fprintf('Theta found by gradient descent:\n');              % 打印结果theta
fprintf('%f\n', theta);

fprintf('Program paused. Press enter to continue.\n\n\n');
pause;
%% ===============================================













%% ====================% 3.绘制y关于x的图像,人口-利润图像===========================
hold on;                        % keep previous plot visible
plot(X(:, 2), X * theta, '-')
legend('Training data', 'Linear regression')
hold off                        % dont overlay any more plots on this figure
%% ===============================================












%% =====================% 4.绘制损失Loss关于theta两个参数的分布图像:==========================
theta0_vals = linspace(-10, 10, 100);   % 创建一个均匀包含100个数字的从-10到10的向量
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)

    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i, j) = computeCost(X, y, t);
    end

end

J_vals = J_vals';                       % 将这个转置一下
% 绘制一个Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)  % 绘制一个三维平面图

xlabel('\theta_0'); 
ylabel('\theta_1');

% 绘制一个Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))  % 绘制一个等高线图
xlabel('\theta_0'); 
ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

%% ===============================================
