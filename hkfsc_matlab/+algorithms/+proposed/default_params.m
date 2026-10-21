function p = default_params()
%DEFAULT_PARAMS Development defaults; formal experiments must explicitly lock them.
p = struct();
p.m = 2;
p.lambda1 = 0.1;
p.lambda2 = 0.01;
p.embedding_rank = [];
p.max_outer_iter = 100;
p.outer_tol = 1e-6;
p.stopping_rule = 'stationarity';
p.min_outer_iter = 20;
p.outer_iterate_tol = 1e-4;
p.convergence_window = 5;
p.membership_admm_steps = 100;
p.membership_pg_steps = 10;
p.membership_pg_tol = 1e-7;
p.rho = 1;
p.adaptive_rho = false;
p.rho_balance_mu = 10;
p.rho_scale = 2;
p.rho_min = 1e-4;
p.rho_max = 1e4;
p.admm_abs_tol = 1e-4;
p.admm_rel_tol = 1e-3;
p.membership_step = 0.1;
p.embedding_step = 0.1;
p.backtrack_factor = 0.5;
p.backtrack_max = 20;
p.verbose = false;
p.kernel = struct('name','HGK','sigma','auto_median','knn',10, ...
    'delta',10,'curvature',1,'normalization','zscore');
end
