function U = project_simplex_rows(Y)
%PROJECT_SIMPLEX_ROWS Euclidean projection of each row onto the probability simplex.
[n, k] = size(Y);
U = zeros(n, k);
for i = 1:n
    y = Y(i, :);
    s = sort(y, 'descend');
    cssv = cumsum(s) - 1;
    rho = find(s - cssv ./ (1:k) > 0, 1, 'last');
    if isempty(rho)
        U(i, :) = ones(1, k) / k;
    else
        theta = cssv(rho) / rho;
        U(i, :) = max(y - theta, 0);
    end
end
end

