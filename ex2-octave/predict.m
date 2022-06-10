function p = predict(theta, X)
    %PREDICT Predict whether the label is 0 or 1 using learned logistic
    %regression parameters theta
    %   p = PREDICT(theta, X) computes the predictions for X using a
    %   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

    m = size(X, 1); % Number of training examples

    % You need to return the following variables correctly
    p = zeros(m, 1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Complete the following code to make predictions using
    %               your learned logistic regression parameters.
    %               You should set p to a vector of 0's and 1's
    %

    % p是m*1
    % sigmoid(theta' * X) > 0.5是阈值
    % 通过下面这个巧妙地转化一下，先都减去0.5，然后就可以判断正负号了，正号的代表1，负号代表0,
    % 再然后，就是将-1和1转化成0和1; 可以先同除0.5变成-0.5和0.5，然后同加0.5，就变成0和1了，完美
    temp = sigmoid(X * theta) - (ones(m, 1) / 2);
    p = (sign(temp) / 2) + (ones(m, 1) / 2);

    % =========================================================================

end
