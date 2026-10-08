function D2 = pairwise_sqdist(A, B)
%PAIRWISE_SQDIST Stable squared Euclidean distances between rows.
D2 = sum(A.^2, 2) + sum(B.^2, 2)' - 2 * (A * B');
D2 = max(D2, 0);
end
