function Xh = map_to_poincare(X, delta, c)
%MAP_TO_POINCARE Radial compression into the Poincare ball of curvature -c.
if delta <= 0 || c <= 0
    error('HKFSC:InvalidGeometry', 'delta and curvature c must be positive.');
end
r = sqrt(sum(X.^2, 2));
Xh = (X ./ (r + delta)) / sqrt(c);
Xh(r == 0, :) = 0;
limit = (1 - 1e-10) / sqrt(c);
rh = sqrt(sum(Xh.^2, 2));
bad = rh >= limit;
if any(bad)
    Xh(bad, :) = Xh(bad, :) .* (limit ./ rh(bad));
end
end
