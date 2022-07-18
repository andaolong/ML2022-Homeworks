function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
    %   cost of using theta as the parameter for linear regression to fit the 
    %   data points in X and y. Returns the cost in J and the gradient in grad

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly 
    J = 0;
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost and gradient of regularized linear 
    %               regression for a particular choice of theta.
    %
    %               You should set J to the cost and grad to the gradient.
    %

    % 根据公式来计算损失J和梯度grad
    % theta: 2*1
    % X: m*2（已经加上了偏置项）
    % y: m*1 

    % 1.计算损失J
    % 计算损失（不带正则化项）
    J = 1/(2*m) * sum((X * theta - y).^2);
    % 加上正则化项
    J = J + lambda/(2*m) * sum(theta(2:end).^2);

    % 计算梯度
    % 计算梯度（先不带正则化项）
    grad = 1/m * sum( (X*theta - y) .* X)';
    % 加上正则化项（上面那行需要转置一下以保证grad和theta形状相同）
    grad(2:end) = grad(2:end) + lambda/m * theta(2:end);
    % =========================================================================
    grad = grad(:);

end
