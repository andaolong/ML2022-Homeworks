function p = predict(Theta1, Theta2, X)
    %PREDICT Predict the label of an input given a trained neural network
    %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    %   trained weights of a neural network (Theta1, Theta2)

    % Useful values
    m = size(X, 1);
    num_labels = size(Theta2, 1);

    % You need to return the following variables correctly 
    p = zeros(size(X, 1), 1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Complete the following code to make predictions using
    %               your learned neural network. You should set p to a 
    %               vector containing labels between 1 to num_labels.
    %
    % Hint: The max function might come in useful. In particular, the max
    %       function can also return the index of the max element, for more
    %       information see 'help max'. If your examples are in rows, then, you
    %       can use max(A, [], 2) to obtain the max for each row.
    %


    % Theta1 has size 25 x 401
    % Theta2 has size 10 x 26

    % 第二层
    a1 = [ones(m, 1) X];    % a1 has size 5000 x 401
    z2 = a1 * Theta1';      % z2 has size 5000 x 25
    a2 = sigmoid(z2);       % 这一步得要，因为题目提供的theta就是带着这一步训练出来的
    a2 = [ones(m, 1) a2];   % z2 has size 5000 x 26

    % 第三层
    z3 = a2 * Theta2';      % z3 has size 5000 x 10
    a3 = sigmoid(z3);       % 

    [_, p] = max(a3, [], 2);
    % h * Theta2' has size 5000 x 10，然后通过max函数取最大数的索引即是答案


    % =========================================================================


end
