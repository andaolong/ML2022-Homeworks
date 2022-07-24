function idx = findClosestCentroids(X, centroids)
    %FINDCLOSESTCENTROIDS computes the centroid memberships for every example
    %   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
    %   in idx for a dataset X where each row is a single example. idx = m x 1 
    %   vector of centroid assignments (i.e. each entry in range [1..K])
    %

    % Set K, 有三个中心点
    K = size(centroids, 1);

    % You need to return the following variables correctly. m*1
    idx = zeros(size(X,1), 1);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Go over every example, find its closest centroid, and store
    %               the index inside idx at the appropriate location.
    %               Concretely, idx(i) should contain the index of the centroid
    %               closest to example i. Hence, it should be a value in the 
    %               range 1..K
    %
    % Note: You can use a for-loop over the examples to compute this.
    % 寻找相对于样本点最近的中心点
    % m*3
    distance = zeros(size(X,1), K);

    for i = 1:size(X, 1)
        % 计算样本点和K个中心点的距离
        for k = 1:K
            distance(i, k) = sum((X(i, :) - centroids(k, :)).^2);
        end
        % idx取使其距离最小的中心点的index
        [_, idx(i)] = min(distance(i, :));
    end
    % =============================================================

end

