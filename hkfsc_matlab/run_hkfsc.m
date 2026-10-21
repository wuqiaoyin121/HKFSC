function model = run_hkfsc(X, K, regime, seed)
%RUN_HKFSC Run one HKFSC initialization with a frozen manuscript profile.
%   Rows of X are samples. K is the requested number of clusters.
%   Ground-truth assignments are neither required nor used.
if nargin < 4 || isempty(seed), seed = 1001; end
if ~isscalar(seed) || ~isfinite(seed) || seed < 0 || seed ~= round(seed)
    error('HKFSC:InvalidSeed', 'Seed must be a nonnegative integer.');
end
setup_hkfsc();
rng(seed, 'twister');
p = hkfsc_paper_params(regime);
model = algorithms.proposed.hkfsc_core(X, K, p);
end
