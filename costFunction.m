function [J, grad] = costFunction(theta, X, y)

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly
    J = 0;
    grad = zeros(size(theta));

    %  J = 1/(2*m) * sum((X*theta - y).^2);
    %  y是100*1，X是100*3，theta是3*1，
    J = 1 / m * sum((-y .* log(sigmoid(X * theta))) - (ones(m, 1) - y) .* log(ones(m, 1) - sigmoid(X * theta)));
    grad = 1 / m * sum((sigmoid(X * theta) - y) .* X);

end

%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% =============================================================
