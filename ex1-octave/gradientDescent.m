function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    m = length(y);                      % number of training examples
    J_history = zeros(num_iters, 1);    % J_history用来记录每次迭代得到的损失,用于后续绘制图像

    for iter = 1:num_iters
        % andaolong add
        % theta是个两行一列的向量,分别是theta0和theta1
        % X是m行2列的向量
        % theta(1) = theta(1) - alpha/m * sum((X*theta - y) .* X(:,1))
        % theta(2) = theta(2) - alpha/m * sum((X*theta - y) .* X(:,2))
        %上面两行需要用一行同时实现
        % 2-1                                  sum[    (m-2 * 2-1) - m-1]   m-2]
        theta = theta - alpha/m * transpose(   sum(    (X*theta - y)   .* X)   );

        J_history(iter) = computeCost(X, y, theta);
    end

end

%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================
    % Save the cost J in every iteration    
   