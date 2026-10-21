function [Xn, state] = normalize_features(X, mode)
%NORMALIZE_FEATURES Label-free feature preprocessing.
if nargin < 2 || isempty(mode), mode = 'zscore'; end
X = double(X);
if any(~isfinite(X(:))), error('HKFSC:NonFiniteData', 'X contains NaN or Inf.'); end
switch lower(mode)
    case 'zscore'
        state.center = mean(X, 1);
        state.scale = std(X, 0, 1);
        state.scale(state.scale < eps) = 1;
        Xn = (X - state.center) ./ state.scale;
    case 'minmax'
        state.center = min(X, [], 1);
        state.scale = max(X, [], 1) - state.center;
        state.scale(state.scale < eps) = 1;
        Xn = (X - state.center) ./ state.scale;
    case 'none'
        state.center = zeros(1, size(X,2));
        state.scale = ones(1, size(X,2));
        Xn = X;
    otherwise
        error('HKFSC:UnknownNormalization', 'Unknown normalization mode: %s', mode);
end
end

