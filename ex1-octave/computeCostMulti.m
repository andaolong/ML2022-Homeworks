% 多变量线性回归，计算损失值
function J = computeCostMulti(X, y, theta)
    m = length(y);
    % 方法1：
    % 多变量和单变量线性回归中，这个函数是一样的
    % 只是theta的维数更多而已
    J = 1/(2*m) * sum((X*theta - y).^2);
    % 方法2：
    % 指导文档中还提出了一种方法，J(θ)是另外一种写法
    % J = 1/(2*m) * (X * theta - y)' * (X * theta - y);
end






%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% =========================================================================


