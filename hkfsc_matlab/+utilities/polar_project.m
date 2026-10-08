function Z = polar_project(A, rank_target)
%POLAR_PROJECT Project a matrix onto the Stiefel manifold Z'*Z=I.
[P, ~, R] = svd(A, 'econ');
Z = P(:, 1:rank_target) * R(:, 1:rank_target)';
end

