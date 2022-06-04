% 特征归一化 


function [X_norm, mu, sigma] = featureNormalize(X)

	% You need to set these values correctly
	% X_norm是源数据; mu是每个feature的平均值;
	% X是m行n列，mu是1行n列，sigma是1行n列
	X_norm = X;
	mu = zeros(1, size(X, 2));
	sigma = zeros(1, size(X, 2));

	mu = mean(X);                       % 按列算每一列的均值,返回的是行向量
	sigma = std(X);                     % 按列算每一列的标准差，返回的是行向量
	% 输出查看一下平均值和标准差
	mu 
	sigma

	for i = 1:length(X_norm)            % 对每一行,都先减去平均值然后除以标准差
		X_norm(i,:) = (X_norm(i,:) - mu) ./ sigma;
	end

end












%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.



% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
% ============================================================