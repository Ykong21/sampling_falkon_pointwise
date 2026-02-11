function out = kr_sampling(N, X, y, X0s, lambda0, opts)
%KR_SAMPLING  MATLAB version of your R kr_sampling(): test-dependent sampling probs.
%
% out = kr_sampling(N, X, y, X0s, lambda0, opts)
%
% Inputs
%   N       : scalar, must equal size(X,1)
%   X       : N x p training covariates
%   y       : N x 1 training responses
%   X0s     : m x p test covariates
%   lambda0 : L x 1 candidate lambdas (BIC tuning on centers)
%   opts    : struct (optional)
%       .kernel            = 'gaussian'|'laplace'|'matern'     (default 'gaussian')
%       .matern_nu         = 0.5|1.5|2.5                      (default 1.5)
%       .clustering        = 'kmeans'|'kmeans_pp'|'kmeans_rp'|'kmedoids' (default 'kmeans')
%       .n_centers         = 200                              (default min(200,N))
%       .kmeans_maxiter    = 50
%       .kmeans_replicates = 3                                (default 3)   NEW
%       .kmeans_start      = 'plus'                           (default 'plus') NEW
%       .rp_dim            = 24
%       .bw_centers_method = 'median'|'detcov'                (default 'median')
%       .bw_scope          = 'centers'|'train_sample'         (default 'centers')
%       .bw_train_sample   = 2000
%       .tune_bw           = false
%       .bw_scales         = [0.25 0.5 1 2]
%       .joint_tune        = false
%       .normalize_cols    = true
%       .return_weights    = false   % if true, also return out.W (k x m)
%       .verbose           = false
%
% Output struct out
%   out.finalprobs : N x m sampling probabilities (if normalize_cols=true, each col sums to 1)
%   out.idx        : N x 1 cluster labels in {1..k}
%   out.centers    : k x p centers
%   out.lambda_opt : scalar selected lambda
%   out.bw_centers : scalar selected bw for center-stage kernel
%   out.tau        : 1/lambda_opt
%   out.W          : k x m center weights (only if opts.return_weights=true)
%   out.timing     : struct with fields cluster, ystar, tune, weights, total
%
% Notes (matches your R semantics):
%   Gaussian: exp(-||x-y||^2 / rho)   (bw = rho, squared-distance scale)
%   Laplace : exp(-||x-y|| / ell)     (bw = ell, distance scale)
%   Matern  : distance/ell with nu in {0.5,1.5,2.5} (bw = ell)

t_all = tic;

if nargin < 6 || isempty(opts), opts = struct(); end
opts = set_defaults(opts, N);

% ---- coerce & checks ----
X = double(X); X0s = double(X0s);
y = double(y(:));
lambda0 = double(lambda0(:));

if size(X,1) ~= N, error('N must equal size(X,1).'); end
if numel(y) ~= N, error('y must have length N.'); end
if size(X0s,2) ~= size(X,2), error('X0s must have same #cols as X.'); end
if isempty(lambda0), error('lambda0 must be non-empty.'); end

m = size(X0s,1);

% =========================
% 1) clustering
% =========================
if opts.verbose, fprintf('[kr_sampling] clustering...\n'); end
t0 = tic;
[centers, idx] = do_clustering(X, opts);
timing.cluster = toc(t0);
k = size(centers,1);

% =========================
% 2) ystar (cluster mean y)
% =========================
if opts.verbose, fprintf('[kr_sampling] computing ystar...\n'); end
t0 = tic;
cnt  = accumarray(idx, 1, [k 1], @sum, 0);
sumy = accumarray(idx, y, [k 1], @sum, 0);
cnt(cnt==0) = 1;
ystar = sumy ./ cnt;
timing.ystar = toc(t0);

% =========================
% 3) base bw + (bw_scale, lambda) tuning by BIC on centers
% =========================
if opts.verbose, fprintf('[kr_sampling] tuning lambda (and optionally bw_scale)...\n'); end
t0 = tic;

if strcmp(opts.bw_scope, 'centers')
    base_bw = bw_from_points(centers, opts.bw_centers_method, opts.kernel);
else
    S = min(opts.bw_train_sample, N);
    sel = randsample(N, S, false);
    base_bw = bw_from_points(X(sel,:), opts.bw_centers_method, opts.kernel);
end

if opts.tune_bw
    scales = opts.bw_scales(:)';
else
    scales = 1;
end

best_score = Inf;
lambda_opt = lambda0(1);
bw_centers = base_bw;

for s = scales
    bw_try = base_bw * s;

    KC = kernel_mat(centers, centers, bw_try, opts.kernel, opts.matern_nu);
    KC = (KC + KC')/2 + 1e-10*eye(k);

    [Q,D] = eig(KC);
    evals = max(diag(D), 0);
    Vy = Q' * ystar;

    for lam = lambda0'
        frac = evals ./ (evals + lam);
        fc = Q * (frac .* Vy);
        resid = ystar - fc;
        res2 = max(sum(resid.^2), realmin);
        df = sum(frac);
        bic = k * log(res2) + log(k) * df;

        if bic < best_score
            best_score = bic;
            lambda_opt = lam;
            bw_centers = bw_try;
        end
    end

    if ~(opts.joint_tune && opts.tune_bw)
        break;
    end
end

timing.tune = toc(t0);

% =========================
% 4) weights/probs
% =========================
if opts.verbose, fprintf('[kr_sampling] computing sampling probs...\n'); end
t0 = tic;

KC = kernel_mat(centers, centers, bw_centers, opts.kernel, opts.matern_nu);
KC = (KC + KC')/2 + 1e-10*eye(k);

tau = 1 / lambda_opt;
A_W = eye(k) + tau * KC;

% Kxc: m x k
Kxc = kernel_mat(X0s, centers, bw_centers, opts.kernel, opts.matern_nu);

% Solve A_W^{-1} * Kxc^T  -> k x m using Cholesky
R = chol(A_W);                 % upper triangular
Kx0s = R \ (R' \ Kxc');        % k x m

W = abs(Kx0s);
cs = sum(W,1); cs(cs<=0) = 1;
W = W ./ cs;                   % normalize each column

finalweights = (k / N) * W;        % k x m
finalprobs = finalweights(idx, :); % N x m

if opts.normalize_cols
    cs2 = sum(finalprobs, 1);
    cs2(~isfinite(cs2) | cs2<=0) = 1;
    finalprobs = finalprobs ./ cs2;
end

timing.weights = toc(t0);
timing.total = toc(t_all);

% =========================
% output
% =========================
out = struct();
out.finalprobs = finalprobs;
out.idx = idx;
out.centers = centers;
out.lambda_opt = lambda_opt;
out.bw_centers = bw_centers;
out.tau = tau;
out.timing = timing;

if opts.return_weights
    out.W = W;
end

end

% =========================
% helpers
% =========================

function opts = set_defaults(opts, N)
def.kernel = 'gaussian';
def.matern_nu = 1.5;
def.clustering = 'kmeans';
def.n_centers = min(200, N);

def.kmeans_maxiter = 50;
def.kmeans_replicates = 3;     % NEW
def.kmeans_start = 'plus';     % NEW ('plus' or 'sample' or numeric init)

def.rp_dim = 24;

def.bw_centers_method = 'median';
def.bw_scope = 'centers';
def.bw_train_sample = 2000;

def.tune_bw = false;
def.bw_scales = [0.25 0.5 1 2];
def.joint_tune = false;

def.normalize_cols = true;
def.return_weights = false;
def.verbose = false;

f = fieldnames(def);
for i = 1:numel(f)
    if ~isfield(opts, f{i}) || isempty(opts.(f{i}))
        opts.(f{i}) = def.(f{i});
    end
end
end

function [centers, idx] = do_clustering(X, opts)
N = size(X,1);
k = min(opts.n_centers, N);

switch opts.clustering
    case 'kmeans'
        [idx, centers] = kmeans(X, k, ...
            'MaxIter', opts.kmeans_maxiter, ...
            'Replicates', opts.kmeans_replicates, ...
            'Start', opts.kmeans_start);

    case 'kmeans_pp'
        [idx, centers] = kmeans(X, k, ...
            'MaxIter', opts.kmeans_maxiter, ...
            'Replicates', opts.kmeans_replicates, ...
            'Start', 'plus');

    case 'kmeans_rp'
        p = size(X,2);
        q = min(opts.rp_dim, p);
        Rr = randn(p, q) / sqrt(q);
        Z = X * Rr;
        [idx, ~] = kmeans(Z, k, ...
            'MaxIter', opts.kmeans_maxiter, ...
            'Replicates', opts.kmeans_replicates, ...
            'Start', opts.kmeans_start);
        centers = zeros(k, p);
        for j = 1:k
            sel = find(idx == j);
            if isempty(sel)
                centers(j,:) = X(randi(N), :);
            else
                centers(j,:) = mean(X(sel,:), 1);
            end
        end

    case 'kmedoids'
        % requires Statistics and Machine Learning Toolbox
        [idx, centers] = kmedoids(X, k);

    otherwise
        error('Unknown clustering: %s', opts.clustering);
end
end

function bw = bw_from_points(Z, method, kernel)
Z = double(Z);
n = size(Z,1);
p = size(Z,2);
if n < 2
    bw = 1;
    return
end

switch method
    case 'detcov'
        C = cov(Z);
        d = det(C);
        if ~isfinite(d) || d <= 0, d = 1; end
        if strcmp(kernel, 'gaussian')
            bw = max(d, 1e-12);            % rho (squared-distance scale)
        else
            bw = max(d^(1/(2*p)), 1e-12);  % ell (distance scale)
        end

    case 'median'
        % No pdist dependency; approximate median distance by sampling pairs
        max_m = 2000;
        if n > max_m
            Z = Z(randsample(n, max_m, false), :);
            n = size(Z,1);
        end

        nPairsMax = 200000;
        nPairs = min(nPairsMax, floor(n*(n-1)/2));
        if nPairs < 1
            md = 1;
        else
            I = randi(n, nPairs, 1);
            J = randi(n, nPairs, 1);
            same = (I == J);
            while any(same)
                J(same) = randi(n, sum(same), 1);
                same = (I == J);
            end
            dif = Z(I,:) - Z(J,:);
            dists = sqrt(sum(dif.^2, 2));
            md = median(dists);
        end

        if ~isfinite(md) || md <= 0, md = 1; end
        if strcmp(kernel, 'gaussian')
            bw = max(md^2, 1e-12);         % rho
        else
            bw = max(md, 1e-12);           % ell
        end

    otherwise
        error('Unknown bw_centers_method: %s', method);
end
end

function K = kernel_mat(A, B, bw, kernel, matern_nu)
A = double(A); B = double(B);
bw = max(bw, 1e-12);

AA = sum(A.^2, 2);                 % nA x 1
BB = sum(B.^2, 2)';                % 1 x nB
D2 = AA + BB - 2*(A * B');
D2(D2 < 0) = 0;

switch kernel
    case 'gaussian'
        K = exp(-D2 / bw);

    case 'laplace'
        D = sqrt(D2);
        K = exp(-D / bw);

    case 'matern'
        D = sqrt(D2);
        ell = bw;
        if matern_nu == 0.5
            K = exp(-D / ell);
        elseif matern_nu == 1.5
            s = sqrt(3) * D / ell;
            K = (1 + s) .* exp(-s);
        else
            s = sqrt(5) * D / ell;
            K = (1 + s + (s.^2)/3) .* exp(-s);
        end

    otherwise
        error('Unknown kernel: %s', kernel);
end
end
