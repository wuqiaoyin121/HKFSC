function smoke_test()
%SMOKE_TEST Exercise the standalone call chain on generated toy data.
setup_hkfsc();
rng(1001, 'twister');
t = linspace(0, 2*pi, 18)';
X = [cos(t)-2, sin(t); cos(t)+2, sin(t)];
p = hkfsc_paper_params('synthetic');
% Only the smoke test uses reduced limits; manuscript profiles are unchanged.
p.max_outer_iter = 25;
p.membership_admm_steps = 20;
model = algorithms.proposed.hkfsc_core(X, 2, p);
assert(numel(model.labels) == size(X,1));
assert(all(isfinite(model.U(:))));
assert(all(isfinite(model.W(:))));
assert(all(isfinite(model.history.objective)));
fprintf('HKFSC smoke test passed: status=%s, iterations=%d.\n', ...
    model.status, model.iterations);
end
