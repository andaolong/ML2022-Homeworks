% 正规方程法，一步求出来了

function [theta] = normalEqn(X, y)
    theta = zeros(size(X, 2), 1);
    theta = pinv(X' * X) * X' * y;
end



% X是m行n+1列，y是m行1列; theta是n行，n+1列
% n+1行m列 * m行n+1列  -->n+1行m列  m行





%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% -------------------------------------------------------------
% ============================================================


