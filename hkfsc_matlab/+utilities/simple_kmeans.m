function [labels, centers] = simple_kmeans(X, K, max_iter)
%SIMPLE_KMEANS Minimal seeded K-means used only inside an independently seeded run.
if nargin < 3, max_iter = 100; end
n = size(X,1);
if K < 2 || K > n, error('HKFSC:InvalidK', 'K must be between 2 and n.'); end
centers = zeros(K,size(X,2));
centers(1,:) = X(randi(n),:);
nearest = utilities.pairwise_sqdist(X, centers(1,:));
for k = 2:K
    total = sum(nearest);
    if total <= eps
        idx = randi(n);
    else
        idx = find(cumsum(nearest) >= rand()*total, 1, 'first');
    end
    centers(k,:) = X(idx,:);
    nearest = min(nearest, utilities.pairwise_sqdist(X, centers(k,:)));
end
labels = ones(n,1);
for iter = 1:max_iter
    D2 = utilities.pairwise_sqdist(X, centers);
    [~, next] = min(D2, [], 2);
    if iter > 1 && isequal(next, labels), break; end
    labels = next;
    for k = 1:K
        take = labels == k;
        if any(take)
            centers(k,:) = mean(X(take,:),1);
        else
            [~,idx] = max(min(D2,[],2));
            centers(k,:) = X(idx,:);
            labels(idx) = k;
        end
    end
end
end

