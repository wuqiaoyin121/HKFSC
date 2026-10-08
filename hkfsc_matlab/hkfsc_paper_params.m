function p = hkfsc_paper_params(regime)
%HKFSC_PAPER_PARAMS Frozen HKFSC settings for the manuscript's main runs.
%   P = HKFSC_PAPER_PARAMS(REGIME) accepts synthetic, low, medium, or high.
%   These values were copied from the official_v6_simple regime locks.
if nargin < 1
    error('HKFSC:MissingRegime', 'Specify synthetic, low, medium, or high.');
end
p = algorithms.proposed.default_params();
p.m = 2;
p.embedding_rank = [];
p.max_outer_iter = 1000;
p.outer_tol = 1e-6;
p.stopping_rule = 'stationarity';
p.min_outer_iter = 20;
p.convergence_window = 5;
p.membership_admm_steps = 100;
p.membership_pg_steps = 10;
p.membership_pg_tol = 1e-7;
p.admm_abs_tol = 1e-4;
p.admm_rel_tol = 1e-3;
p.membership_step = 0.1;
p.embedding_step = 0.1;
p.backtrack_factor = 0.5;
p.backtrack_max = 20;
p.kernel.name = 'HGK';
p.kernel.sigma = 'auto_median';
p.kernel.delta = 10;
p.kernel.curvature = 1;
p.kernel.normalization = 'minmax';
switch lower(char(regime))
    case 'synthetic'
        p.lambda1 = 10; p.lambda2 = 1e-4; p.kernel.knn = 10;
        p.rho = 1; p.adaptive_rho = true;
        p.outer_iterate_tol = 1e-4;
    case 'low'
        p.lambda1 = 10; p.lambda2 = 1e-5; p.kernel.knn = 15;
        p.rho = 0.1; p.adaptive_rho = false;
        p.outer_iterate_tol = 1e-4;
    case 'medium'
        p.lambda1 = 100; p.lambda2 = 1e-4; p.kernel.knn = 15;
        p.rho = 0.1; p.adaptive_rho = false;
        p.outer_iterate_tol = 1e-4;
    case 'high'
        p.lambda1 = 100; p.lambda2 = 1e-2; p.kernel.knn = 15;
        p.rho = 0.1; p.adaptive_rho = false;
        p.outer_iterate_tol = 3e-4;
    otherwise
        error('HKFSC:UnknownRegime', ...
            'Regime must be synthetic, low, medium, or high.');
end
end
