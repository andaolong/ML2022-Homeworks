function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    %neural network which performs classification
    %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    %   X, y, lambda) computes the cost and gradient of the neural network. The
    %   parameters for the neural network are "unrolled" into the vector
    %   nn_params and need to be converted back into the weight matrices. 
    % 
    %   The returned parameter grad should be a "unrolled" vector of the
    %   partial derivatives of the neural network.
    %

    %input_layer_size  = 400;  % 20x20 Input Images of Digits
    % hidden_layer_size = 25;   % 25 hidden units
    % num_labels = 10;          % 10 labels, from 1 to 10   
    %                           % (note that we have mapped "0" to label 10)

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    % 先将被展开成向量的参数恢复回矩阵形式
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);
            
    % You need to return the following variables correctly 
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));


    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the code by working through the
    %               following parts.
    %


    % 自定义临时变量
    % X新增一列偏置项
    X_temp = [ones(m, 1), X];
















    % Part 1: Feedforward the neural network and return the cost in the
    %         variable J. After implementing Part 1, you can verify that your
    %         cost function computation is correct by verifying the cost
    %         computed in ex4.m
    % 首先，实现前向传播计算出损失值J（从前向后算就行）

    % 可以先用for循环实现一下
    % 将y从标签转化为矩阵形式
    % y_new = zeros(m, num_labels);
    % for i = 1:m
    %     y_new(i, y(i)) = 1;

    % % 这样for循环会很慢。。还是用向量比较快
    % for i = 1:m
    %     for k = 1:num_labels
    %         % 先通过前向传播计算出估计值h
    %         a1 = X_temp(i, :);      % 1*401
    %         z2 = Theta1 * a1';      % 25*401 x 401*1 = 25*1
    %         a2 = sigmoid(z2);       % 25*1
    %         a2 = [1; a2];           % 26*1
    %         z3 = Theta2 * a2;       % 10*26 x 26*1 = 10*1
    %         a3 = sigmoid(z3);
    %         h = a3;                 % 10*1
    %         [_, hIndex] = max(h, [], 1); 
    %         % 然后计算损失
    %         % h(j)是第k个输出单元的激活值
    %         J_temp = -y_new(i, hIndex) * log(h(k)) - (1 - y_new(i, hIndex)) * log(1 - h(k));
    %         J = J + J_temp;
    %     endfor
    % endfor
    % J = J / m;
    
    % 向量实现
    % 将y从标签转化为矩阵形式
    y_new = zeros(m, num_labels);   % 5000*10
    for i = 1:m
        y_new(i, y(i)) = 1;
    endfor
    % 开始进行前向传播
    a1 = X_temp;            % 5000*401
    z2 = a1 * Theta1';      % 5000*401 x 401*25 = 5000*25
    a2 = sigmoid(z2);       % 5000*25
    a2 = [ones(m,1), a2];   % 5000*26
    z3 = a2 * Theta2';      % 5000*26 x 26*10 = 5000*10
    a3 = sigmoid(z3);
    h = a3;                 % 5000*10

    % 5000*10进行两次sum，一次是对10就是k，一次是对5000就是m
    J = sum(sum( -y_new.*log(h) - (1-y_new).*log(1-h))) / m;
    % 到这里损失计算出来是0.287629，和预期结果吻合


    % 添加正则化项
    % Theta1是25*401 Theta2是10*26，算正则化项的时候不能算第一个
    % J = J + lambda/(2*m) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));
    % 0.383770,符合预期 
    J = J + lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));


















    % Part 2: Implement the backpropagation algorithm to compute the gradients
    %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    %         Theta2_grad, respectively. After implementing Part 2, you can check
    %         that your implementation is correct by running checkNNGradients
    %
    %         Note: The vector y passed into the function is a vector of labels
    %               containing values from 1..K. You need to map this vector into a 
    %               binary vector of 1's and 0's to be used with the neural network
    %               cost function.
    %               就是说，记得把提供的标注向量y([1,2,7,2....])映射到二进制向量[[10000][000100]]中去
    %
    %         Hint: We recommend implementing backpropagation using a for-loop
    %               over the training examples if you are implementing it for the 
    %               first time.
    %
    delta_accumulate_1 = zeros(size(Theta1)); 
    delta_accumulate_2 = zeros(size(Theta2));
    for t = 1:m
        % 这里的a和z都是一个样本的
        % step1：前向传播得到z和a
        % X_temp已经加上偏置bias了
        a_1 = X_temp(t, :)';    % 401*1
        
        z_2 = Theta1 * a_1;     % 25*401 x 401*1 = 25*1
        a_2 = sigmoid(z_2);
        a_2 = [1; a_2];         % 26*1

        z_3 = Theta2 * a_2;     % 10*26 x 26*1 = 10*1
        a_3 = sigmoid(z_3);

        % step2: 对每个unit k求delta
        delta_3 = a_3 - y_new(t, :)';    % 10*1

        % step3: 对第二层求delta 
        % (26*10 x 10*1) .* (26*1)
        delta_2 = (Theta2' * delta_3) .* sigmoidGradient([1; z_2]);
        delta_2 = delta_2(2:end);   % remove delta_0

        % step4: 累计此示例中的梯度; 这里delta项去掉偏置项，a不去偏置项
        % delta_accumulate_2 = 10*1 * 1*26 = 10*26
        % delta_accumulate_1 = 25*1 * 1*401 = 25*401
        delta_accumulate_2 = delta_accumulate_2 + delta_3 * a_2';
        delta_accumulate_1 = delta_accumulate_1 + delta_2 * a_1';  
    endfor

    % step5: 得到梯度项
    Theta1_grad = 1/m * delta_accumulate_1;     % 10*26 
    Theta2_grad = 1/m * delta_accumulate_2;     % 25*401




    % Part 3: Implement regularization with the cost function and gradients.
    %
    %         Hint: You can implement this around the code for
    %               backpropagation. That is, you can compute the gradients for
    %               the regularization separately and then add them to Theta1_grad
    %               and Theta2_grad from Part 2.
    %
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);

    % -------------------------------------------------------------

    % =========================================================================

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
