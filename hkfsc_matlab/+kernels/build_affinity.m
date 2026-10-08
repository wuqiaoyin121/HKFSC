function [W, L, info] = build_affinity(X, params)
%BUILD_AFFINITY Build one fixed nonnegative hyperbolic affinity graph.
defaults = struct('name','HGK','sigma','auto_median','knn',10, ...
    'delta',10,'curvature',1,'normalization','zscore');
p = utilities.with_defaults(params, defaults);
[Xn, prep] = utilities.normalize_features(X, p.normalization);
if strcmpi(p.name,'EGAUSS')
    Xh = [];
    D = sqrt(max(utilities.pairwise_sqdist(Xn,Xn),0));
    geometry = 'euclidean';
else
    Xh = kernels.map_to_poincare(Xn, p.delta, p.curvature);
    D = pair_dissimilarity(Xh, upper(p.name), p.curvature);
    geometry = 'poincare_ball';
end
n = size(D, 1);
D(1:n+1:end) = inf;

if ischar(p.sigma) || isstring(p.sigma)
    if ~strcmpi(string(p.sigma), "auto_median")
        error('HKFSC:InvalidSigma', 'Unknown sigma rule: %s', string(p.sigma));
    end
    k0 = min(max(1, p.knn), n-1);
    sorted = sort(D, 2, 'ascend');
    pool = sorted(:, 1:k0);
    pool = pool(isfinite(pool) & pool > 0);
    if isempty(pool), sigma = 1; else, sigma = median(pool); end
else
    sigma = double(p.sigma);
end
if ~isscalar(sigma) || ~isfinite(sigma) || sigma <= 0
    error('HKFSC:InvalidSigma', 'Resolved kernel bandwidth must be positive.');
end

switch upper(p.name)
    case {'HGK','MGAUSS','IPRK','ILRK','EGAUSS'}
        A = exp(-(D.^2) ./ (2*sigma^2));
    case {'HLAP','MLAP','IPLK','ILLK'}
        A = exp(-D ./ sigma);
    otherwise
        error('HKFSC:UnknownKernel', 'Unknown kernel: %s', p.name);
end
A(~isfinite(A)) = 0;
A(1:n+1:end) = 0;

k = min(max(1, round(p.knn)), n-1);
[~, order] = sort(D, 2, 'ascend');
mask = false(n);
for i = 1:n
    mask(i, order(i,1:k)) = true;
end
A(~mask) = 0;
W = 0.5 * (A + A');
W(W < 0 | ~isfinite(W)) = 0;
W(1:n+1:end) = 0;

degree = sum(W, 2);
invroot = 1 ./ sqrt(max(degree, eps));
L = eye(n) - (invroot .* W) .* invroot';
L = 0.5 * (L + L');

info = struct('kernel',upper(p.name),'sigma',sigma,'knn',k, ...
    'delta',p.delta,'curvature',p.curvature,'normalization',p.normalization, ...
    'geometry',geometry,'preprocessing',prep,'X_hyperbolic',Xh,'pair_dissimilarity',D, ...
    'degree',degree,'isolated_nodes',find(degree <= eps));
end

function D = pair_dissimilarity(X, name, c)
n = size(X, 1);
D = zeros(n);
for i = 1:n-1
    for j = i+1:n
        x = X(i,:); y = X(j,:);
        switch name
            case {'HGK','HLAP'}
                v = mobius_add(-x, y, c);
                nv = min(sqrt(c) * norm(v), 1-1e-12);
                d = 2 * atanh(nv) / sqrt(c);
            case {'MGAUSS','MLAP'}
                d = norm(mobius_add(-x, y, c));
            case {'IPRK','IPLK'}
                fij = poincare_log(x, y, c);
                fji = poincare_log(y, x, c);
                d = norm(fij - fji);
            case {'ILRK','ILLK'}
                lx = poincare_to_lorentz(x, c);
                ly = poincare_to_lorentz(y, c);
                fij = lorentz_log(lx, ly, c);
                fji = lorentz_log(ly, lx, c);
                d = norm(fij - fji);
            otherwise
                error('HKFSC:UnknownKernel', 'Unknown kernel: %s', name);
        end
        if ~isfinite(d), d = realmax('double')^(1/4); end
        D(i,j) = d; D(j,i) = d;
    end
end
end

function z = mobius_add(x, y, c)
xy = dot(x,y); x2 = dot(x,x); y2 = dot(y,y);
den = 1 + 2*c*xy + c^2*x2*y2;
den = max(den, eps);
z = ((1 + 2*c*xy + c*y2)*x + (1-c*x2)*y) / den;
end

function v = poincare_log(x, y, c)
z = mobius_add(-x, y, c);
nz = norm(z);
if nz < eps, v = zeros(size(x)); return; end
lambda_x = 2 / max(1-c*dot(x,x), eps);
arg = min(sqrt(c)*nz, 1-1e-12);
v = (2/(sqrt(c)*lambda_x)) * atanh(arg) * z/nz;
end

function l = poincare_to_lorentz(x, c)
r2 = dot(x,x); den = max(1-c*r2, eps);
l = [(1+c*r2)/(sqrt(c)*den), 2*x/den];
end

function v = lorentz_log(x, y, c)
ip = -x(1)*y(1) + dot(x(2:end), y(2:end));
a = max(-c*ip, 1);
d = acosh(a);
if d < 1e-12
    v = zeros(size(x));
else
    v = (d/sinh(d)) * (y + c*ip*x);
end
end
