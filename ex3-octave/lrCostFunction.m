function [J, grad] = lrCostFunction(theta, X, y, lambda)
    %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization
    %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly 
    J = 0;
    grad = zeros(size(theta));

    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Hint: The computation of the cost function and gradients can be
    %       efficiently vectorized. For example, consider the computation
    %
    %           sigmoid(X * theta)
    %
    %       Each row of the resulting matrix will contain the value of the
    %       prediction for that example. You can make use of this to vectorize
    %       the cost function and gradient computations. 
    %
    % Hint: When computing the gradient of the regularized cost function, 
    %       there're many possible vectorized solutions, but one solution
    %       looks like:
    %           grad = (unregularized gradient for logistic regression)
    %           temp = theta; 
    %           temp(1) = 0;   % because we don't add anything for j = 0  
    %           grad = grad + YOUR_CODE_HERE (using the temp variable)
    %

    % 这个函数也需要加上正则化项
    theta1 = theta;
    theta1(1) = 0;  
    % 这个第一项赋值为0是因为计算损失时，sum求和是从1到m，没有0，所有直接theta1平方就可以
    % 啊，这个将theta(0)赋值为1的theta1是真的好用呜呜呜
    % 然后以后我写这种复杂公式的时候也要注意分步，不然以后读的时候太痛苦了。。。


    h = sigmoid(X * theta)
    % h = (1.0)./(1.0+exp(-X*theta));

    %fprintf('y and h\n');
    %size(y)
    %size(h)

    J = (1.0/m)*sum(log(h')*-y - log(1-h')*(1-y));
    J = J + (lambda/(2.0*m))*theta1'*theta1;

    d = h-y;
    grad = (1.0/m)*(X'*d);
    grad = grad + (lambda/m) * theta1;

    % =============================================================

    grad = grad(:);

end
