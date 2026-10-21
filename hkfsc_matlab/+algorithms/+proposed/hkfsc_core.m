function model = hkfsc_core(X, K, params)
%HKFSC_CORE Canonical fixed-graph HKFSC optimizer for Eqs. (40)-(60).
if nargin < 3, params = struct(); end
p = utilities.with_defaults(params, algorithms.proposed.default_params());
validate_inputs(X, K, p);
[W, L, graph] = kernels.build_affinity(X, p.kernel);
n = size(X,1);
r = p.embedding_rank;
if isempty(r), r = K; end
if r > min(n,K), error('HKFSC:InvalidRank', 'embedding_rank must not exceed min(n,K).'); end

[Z,init_info] = spectral_initialization(L,r);
[init_labels,V] = utilities.simple_kmeans(Z,K,50);
D2 = utilities.pairwise_sqdist(Z,V);
U = fuzzy_from_distances(D2,p.m);
if any(~isfinite(U(:)))
    U = full(sparse((1:n)',init_labels,1,n,K));
end
Q = L' * Z;

[ei,ej,ew] = find(triu(W,1));
E = numel(ew);
H = sparse([(1:E)';(1:E)'],[ei;ej],[ones(E,1);-ones(E,1)],E,n);
T = zeros(E,K);
Y = zeros(E,K);

history = nan(p.max_outer_iter,14);
status = 'max_iter';
convergence_streak = 0;
rho = p.rho;
for outer = 1:p.max_outer_iter
    previous = objective_terms(L,Z,Q,U,V,H,ew,p);
    previous_U = U;
    previous_projector = Z*Z';
    [U,T,Y,primal,dual,eps_pri,eps_dual,admm_used,rho] = ...
        update_membership(U,T,Y,H,ew,Z,V,p,rho);
    Um = U.^p.m;
    denom = sum(Um,1)';
    V = (Um' * Z) ./ max(denom,eps);
    [Z,zstep] = update_embedding(Z,L,Q,U,V,p);
    Q = L' * Z;
    current = objective_terms(L,Z,Q,U,V,H,ew,p);
    rel = abs(previous.total-current.total)/max(1,abs(previous.total));
    membership_change = norm(U-previous_U,'fro')/max(1,norm(previous_U,'fro'));
    current_projector = Z*Z';
    subspace_change = norm(current_projector-previous_projector,'fro')/ ...
        max(1,norm(previous_projector,'fro'));
    admm_ok = primal <= eps_pri && dual <= eps_dual;
    switch lower(string(p.stopping_rule))
        case "legacy_objective"
            % official_v5 behavior: objective change plus a solved ADMM block.
            stable = rel < p.outer_tol && admm_ok;
        case "stationarity"
            stable = rel < p.outer_tol && membership_change < p.outer_iterate_tol && ...
                subspace_change < p.outer_iterate_tol && admm_ok;
        otherwise
            error('HKFSC:StoppingRule','Unknown stopping rule: %s.',p.stopping_rule);
    end
    if stable,convergence_streak=convergence_streak+1;else,convergence_streak=0;end
    history(outer,:) = [current.total,current.reconstruction,current.fuzzy, ...
        current.mal,primal,dual,eps_pri,eps_dual,admm_used,zstep, ...
        membership_change,subspace_change,convergence_streak,rho];
    if p.verbose
        fprintf('HKFSC iter %d: obj %.8g, rel %.3g, ADMM r %.3g s %.3g\n', ...
            outer,current.total,rel,primal,dual);
    end
    if ~isfinite(current.total)
        status = 'nonfinite_objective';
        break;
    end
    if outer >= p.min_outer_iter && convergence_streak >= p.convergence_window
        status = 'converged';
        break;
    end
end
history = history(1:outer,:);
[~,labels] = max(U,[],2);
model = struct('labels',labels,'U',U,'Z',Z,'V',V,'Q',Q,'W',W,'L',L, ...
    'graph_info',graph,'initialization_info',init_info,'params',p, ...
    'status',status,'iterations',outer, ...
    'history',array2table(history,'VariableNames',{'objective','reconstruction', ...
    'fuzzy','mal','admm_primal','admm_dual','admm_eps_primal', ...
    'admm_eps_dual','admm_iterations','embedding_step','membership_change', ...
    'subspace_change','convergence_streak','admm_rho'}));
end

function [Z,info] = spectral_initialization(L,r)
% Use the r smoothest eigenvectors of the fixed normalized Laplacian.
n=size(L,1);method='eigs_smallestreal';eigs_flag=NaN;
try
    opts=struct('tol',1e-10,'maxit',max(300,5*n),'disp',0);
    [Z,D,eigs_flag]=eigs(L,r,'smallestreal',opts);
    eigenvalues=real(diag(D));Z=real(Z);
    if eigs_flag~=0||numel(eigenvalues)~=r||any(~isfinite(eigenvalues))||any(~isfinite(Z(:)))
        error('HKFSC:EigsNotConverged', ...
            'Iterative spectral initialization returned an incomplete or nonfinite eigensystem.');
    end
catch
    method='full_eig_fallback';
    [all_vectors,all_values]=eig(full(L),'vector');
    [eigenvalues,order]=sort(real(all_values),'ascend');
    eigenvalues=eigenvalues(1:r);Z=real(all_vectors(:,order(1:r)));
end
if numel(eigenvalues)~=r||any(~isfinite(eigenvalues))||any(~isfinite(Z(:)))
    error('HKFSC:SpectralInitializationFailed', ...
        'Spectral initialization failed to produce %d finite eigenpairs.',r);
end
[~,order]=sort(eigenvalues,'ascend');
eigenvalues=eigenvalues(order);Z=Z(:,order);
Z=utilities.polar_project(Z,r);
info=struct('method',method,'eigenvalues',eigenvalues(:)', ...
    'eigs_flag',eigs_flag,'orthogonality_residual',norm(Z'*Z-eye(r),'fro'), ...
    'eigen_residual',norm(L*Z-Z*diag(eigenvalues),'fro'), ...
    'source','fixed_normalized_hyperbolic_graph_laplacian');
end

function validate_inputs(X,K,p)
if ~ismatrix(X) || size(X,1) < 3 || any(~isfinite(X(:)))
    error('HKFSC:InvalidData','X must be a finite n-by-d matrix with n >= 3.');
end
if K < 2 || K >= size(X,1) || K ~= round(K)
    error('HKFSC:InvalidK','K must be an integer in [2,n-1].');
end
if p.m <= 1 || p.lambda1 < 0 || p.lambda2 < 0 || p.rho <= 0
    error('HKFSC:InvalidParameters','Require m>1, lambda1/lambda2>=0, and rho>0.');
end
if p.min_outer_iter < 2 || p.min_outer_iter ~= round(p.min_outer_iter) || ...
        p.convergence_window < 1 || p.convergence_window ~= round(p.convergence_window) || ...
        p.outer_iterate_tol <= 0
    error('HKFSC:InvalidStoppingRule', ...
        'Require integer min_outer_iter>=2, integer convergence_window>=1, and outer_iterate_tol>0.');
end
if ~ismember(lower(string(p.stopping_rule)),["legacy_objective","stationarity"])
    error('HKFSC:InvalidStoppingRule','Use legacy_objective or stationarity.');
end
if ~isscalar(p.adaptive_rho) || ...
        ~ismember(double(p.adaptive_rho),[0 1]) || ...
        p.rho_balance_mu <= 1 || p.rho_scale <= 1 || ...
        p.rho_min <= 0 || p.rho_max < p.rho_min || ...
        p.rho < p.rho_min || p.rho > p.rho_max
    error('HKFSC:InvalidAdaptiveRho', ...
        'Invalid adaptive-rho settings. Require mu/tau>1 and 0<rho_min<=rho_max.');
end
end

function U = fuzzy_from_distances(D2,m)
n = size(D2,1); K = size(D2,2);
U = zeros(n,K);
zero = D2 <= 1e-14;
for i=1:n
    if any(zero(i,:))
        U(i,zero(i,:)) = 1/sum(zero(i,:));
    else
        q = D2(i,:).^(-1/(m-1));
        U(i,:) = q/sum(q);
    end
end
end

function [U,T,Y,primal,dual,eps_pri,eps_dual,used,rho] = ...
        update_membership(U,T,Y,H,ew,Z,V,p,rho)
D2 = utilities.pairwise_sqdist(Z,V);
E = size(H,1); K = size(U,2);
if E == 0
    error('HKFSC:EmptyGraph','Affinity graph has no edges.');
end
% With lambda2=0 the membership block has the exact FCM solution.  This
% makes the w/o-MAL ablation identical to the full model except for MAL.
if p.lambda2==0
    U=fuzzy_from_distances(D2,p.m);
    T=H*U;Y=zeros(size(T));primal=0;dual=0;eps_pri=0;eps_dual=0;used=0;
    return;
end
primal=inf; dual=inf;eps_pri=inf;eps_dual=inf;
for used=1:p.membership_admm_steps
    Tprev=T;
    DU=H*U;
    T=soft_threshold(DU+Y/rho,(p.lambda2/rho)*ew);
    % Solve the convex U subproblem to its own tolerance before the dual
    % update.  A single projected-gradient step caused the old ADMM loop to
    % stall at nonzero primal and dual residuals.
    for pg=1:p.membership_pg_steps
        grad = p.lambda1*p.m*(max(U,0).^(p.m-1)).*D2 + H'*(Y+rho*(H*U-T));
        step=p.membership_step;base=augmented_u(U,D2,H,T,Y,p,rho);
        accepted=false;
        for bt=1:p.backtrack_max
            candidate=utilities.project_simplex_rows(U-step*grad);
            if augmented_u(candidate,D2,H,T,Y,p,rho) <= base+1e-12
                accepted=true;break;
            end
            step=step*p.backtrack_factor;
        end
        if ~accepted
            error('HKFSC:MembershipLineSearch','Membership line search failed to find a descent step.');
        end
        relative_change=norm(candidate-U,'fro')/max(1,norm(U,'fro'));
        U=candidate;
        if relative_change<=p.membership_pg_tol,break;end
    end
    residual=H*U-T;
    Y=Y+rho*residual;
    primal=norm(residual,'fro');
    dual=rho*norm(H'*(T-Tprev),'fro');
    eps_pri=sqrt(E*K)*p.admm_abs_tol+p.admm_rel_tol*max(norm(H*U,'fro'),norm(T,'fro'));
    eps_dual=sqrt(size(U,1)*K)*p.admm_abs_tol+p.admm_rel_tol*norm(H'*Y,'fro');
    if primal<=eps_pri && dual<=eps_dual, break; end
    if p.adaptive_rho
        if primal > p.rho_balance_mu*dual
            rho=min(p.rho_max,rho*p.rho_scale);
        elseif dual > p.rho_balance_mu*primal
            rho=max(p.rho_min,rho/p.rho_scale);
        end
    end
end
end

function T=soft_threshold(A,tau)
T=sign(A).*max(abs(A)-tau,0);
end

function value=augmented_u(U,D2,H,T,Y,p,rho)
R=H*U-T;
value=p.lambda1*sum(sum((U.^p.m).*D2))+sum(sum(Y.*R))+0.5*rho*sum(R(:).^2);
end

function [Z,step]=update_embedding(Z,L,Q,U,V,p)
Um=U.^p.m;
degree_u=sum(Um,2);
grad=-2*L*Q+2*Z*(Q'*Q)+2*p.lambda1*(degree_u.*Z-Um*V);
base=embedding_value(Z,L,Q,Um,V,p.lambda1);
step=p.embedding_step;
for bt=1:p.backtrack_max
    candidate=utilities.polar_project(Z-step*grad,size(Z,2));
    if embedding_value(candidate,L,Q,Um,V,p.lambda1)<=base+1e-12, break; end
    step=step*p.backtrack_factor;
end
Z=candidate;
end

function value=embedding_value(Z,L,Q,Um,V,lambda1)
R=L-Z*Q';
D2=utilities.pairwise_sqdist(Z,V);
value=sum(R(:).^2)+lambda1*sum(sum(Um.*D2));
end

function out=objective_terms(L,Z,Q,U,V,H,ew,p)
R=L-Z*Q';
out.reconstruction=sum(R(:).^2);
D2=utilities.pairwise_sqdist(Z,V);
out.fuzzy=p.lambda1*sum(sum((U.^p.m).*D2));
DU=H*U;
out.mal=p.lambda2*sum(ew.*sum(abs(DU),2));
out.total=out.reconstruction+out.fuzzy+out.mal;
end
