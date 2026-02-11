function out = sampling_falkon_pointwise(Xtr, ytr, Xte, varargin)
% sampling_falkon_pointwise
%
% Core pipeline (dataset-agnostic):
%   For each test point x0 = Xte(e,:):
%     1) Compute pointwise sampling probabilities p_e over training points using kr_sampling()
%     2) Sample n_sub points with replacement using p_e
%     3) Train FALKON on the sampled subset (pointwise training)
%     4) Predict yhat(x0)
%
% INPUTS
%   Xtr : N x p training covariates
%   ytr : N x 1 training response (raw scale)
%   Xte : m x p test covariates (can be 1 x p)
%
% NAME-VALUE OPTIONS (defaults are sensible; override as needed)
%   'standardizeX'   : true/false (default true)  standardize X using train mean/std
%   'standardizeY'   : true/false (default true)  standardize y for kr_sampling + falkon training
%
%   'lambda_grid'    : vector of candidate lambdas for kr_sampling (default 10.^linspace(-2,4,30))
%   'kr_opts'        : struct passed to kr_sampling (default minimal gaussian + kmeans config)
%
%   'n_sub'          : resample size (default 10000)
%   'M_centers'      : FALKON centers M (default 500, uses min(M, n_sub))
%   'lambda_falkon'  : FALKON ridge lambda (default 1e-6)
%   'falkon_iters'   : FALKON iterations (default 20)
%   'useGPU'         : 0/1 passed to falkon (default 0)
%   'memToUseGB'     : memory passed to falkon (default 8)
%
%   'rbf_sigma'      : sigma for RBF kernel used in FALKON (default 6)
%   'kernel_fn'      : custom kernel handle @(A,B)K(A,B) (optional). If provided, overrides rbf_sigma.
%
%   'rng_base'       : base seed (default 1000) deterministic per test point
%   'store_probs'    : true/false store probs for each test (default false)
%   'verbose'        : true/false (default true)
%
% OUTPUT (struct)
%   out.yhat         : m x 1 predictions on raw y scale
%   out.train_time_s : m x 1 FALKON training time for each test point
%   out.idx_sub      : cell(m,1) sampled training indices (with replacement)
%   out.idx_centers  : cell(m,1) chosen centers indices within subset
%   out.probs        : (optional) cell(m,1) probability vector over N (can be huge)
%   out.kr           : kr_sampling returned struct (contains centers, idx, bw, etc.)
%   out.preproc      : preprocessing info (mu/sd)
%
% REQUIREMENTS
%   - kr_sampling.m on path
%   - falkon.m + deps on path
%
% Note: This is the "core pipeline" only. No dataset IO, no uniform baseline,
%       no figures. Build demos on top of this.

% -------------------- Parse args --------------------
p = inputParser;
p.addRequired('Xtr', @(x) isnumeric(x) && ismatrix(x));
p.addRequired('ytr', @(x) isnumeric(x) && (isvector(x) || size(x,2)==1));
p.addRequired('Xte', @(x) isnumeric(x) && ismatrix(x));

p.addParameter('standardizeX', true, @(x) islogical(x) || isnumeric(x));
p.addParameter('standardizeY', true, @(x) islogical(x) || isnumeric(x));

p.addParameter('lambda_grid', 10.^linspace(-2,4,30), @(x) isnumeric(x) && isvector(x));
p.addParameter('kr_opts', default_kr_opts(), @(s) isstruct(s));

p.addParameter('n_sub', 10000, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('M_centers', 500, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('lambda_falkon', 1e-6, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('falkon_iters', 20, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('useGPU', 0, @(x) isnumeric(x) && isscalar(x));
p.addParameter('memToUseGB', 8, @(x) isnumeric(x) && isscalar(x) && x>0);

p.addParameter('rbf_sigma', 6, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('kernel_fn', [], @(f) isempty(f) || isa(f,'function_handle'));

p.addParameter('rng_base', 1000, @(x) isnumeric(x) && isscalar(x));
p.addParameter('store_probs', false, @(x) islogical(x) || isnumeric(x));
p.addParameter('verbose', true, @(x) islogical(x) || isnumeric(x));

p.parse(Xtr, ytr, Xte, varargin{:});
opt = p.Results;

Xtr = double(Xtr);
ytr = double(ytr(:));
Xte = double(Xte);

N = size(Xtr,1);
m = size(Xte,1);

% -------------------- Preprocess --------------------
pre = struct();

% X standardization (train-based)
if opt.standardizeX
    [Xtr_z, muX, sX] = zscore_train_local(Xtr);
    Xte_z = zscore_apply_local(Xte, muX, sX);
    pre.muX = muX; pre.sX = sX;
else
    Xtr_z = Xtr; Xte_z = Xte;
    pre.muX = []; pre.sX = [];
end

% y standardization (train-based)
if opt.standardizeY
    muy = mean(ytr); sdy = std(ytr); if sdy==0, sdy=1; end
    ytr_z = (ytr - muy) / sdy;
    pre.muy = muy; pre.sdy = sdy;
else
    ytr_z = ytr;
    pre.muy = 0; pre.sdy = 1;
end

% kernel handle for FALKON
if isempty(opt.kernel_fn)
    gamma = 1/(2*opt.rbf_sigma^2);
    Kfun = @(A,B) rbf_kernel_local(A,B,gamma);
else
    Kfun = opt.kernel_fn;
end

% -------------------- 1) kr_sampling (compute probs for all test points) --------------------
t_sel = tic;
kr = kr_sampling(N, Xtr_z, ytr_z, Xte_z, opt.lambda_grid, opt.kr_opts);
t_sel_total = toc(t_sel);

finalprobs = kr.finalprobs;  % N x m
% sanity: each col sums to 1 (within fp error)
colsum = sum(finalprobs,1);

if opt.verbose
    fprintf('[PIPE] kr_sampling done. kernel=%s, lambda_opt=%.3g, bw_centers=%.3g, time=%.2fs\n', ...
        opt.kr_opts.kernel, kr.lambda_opt, kr.bw_centers, t_sel_total);
    fprintf('[PIPE] probs colsum range = [%.8f, %.8f]\n', min(colsum), max(colsum));
end

% -------------------- 2) pointwise train+predict --------------------
yhat = zeros(m,1);
train_time_s = zeros(m,1);
idx_sub = cell(m,1);
idx_centers = cell(m,1);

if opt.store_probs
    probs_store = cell(m,1);
else
    probs_store = [];
end

for e = 1:m
    rng(opt.rng_base + e, 'twister');

    probs = finalprobs(:,e);

    % with-replacement sample from full train pool using pointwise probs
    idx_e = randsample(N, opt.n_sub, true, probs);
    Xsub = Xtr_z(idx_e,:);
    ysub = ytr_z(idx_e);

    % choose centers from the subset (without replacement)
    M_eff = min(opt.M_centers, opt.n_sub);
    idx_c = randsample(opt.n_sub, M_eff, false);
    Csub = Xsub(idx_c,:);

    % train falkon (pointwise)
    tt = tic;
    alpha = falkon(Xsub, Csub, Kfun, ysub, opt.lambda_falkon, opt.falkon_iters, ...
        [], @(a,c)[], opt.memToUseGB, opt.useGPU);
    train_time_s(e) = toc(tt);

    % predict at x0
    k0 = Kfun(Xte_z(e,:), Csub);
    yhat_z = k0 * alpha;

    % back to raw y scale (if standardizedY)
    yhat(e) = pre.muy + pre.sdy * yhat_z;

    idx_sub{e} = idx_e;
    idx_centers{e} = idx_c;

    if opt.store_probs
        probs_store{e} = probs; %#ok<AGROW>
    end

    if opt.verbose && (e==1 || mod(e,50)==0 || e==m)
        fprintf('[PIPE] e=%d/%d | train=%.3fs\n', e, m, train_time_s(e));
    end
end

% -------------------- Pack output --------------------
out = struct();
out.yhat = yhat;
out.train_time_s = train_time_s;
out.idx_sub = idx_sub;
out.idx_centers = idx_centers;
out.kr = kr;
out.preproc = pre;
out.select_time_s = t_sel_total;

if opt.store_probs
    out.probs = probs_store;
end

end

% ==================== Defaults + helpers ====================

function opts = default_kr_opts()
opts = struct();
opts.kernel = 'gaussian';
opts.clustering = 'kmeans';
opts.n_centers = 500;
opts.bw_centers_method = 'median';
opts.bw_scope = 'train_sample';
opts.bw_train_sample = 2000;
opts.tune_bw = true;
opts.joint_tune = true;
opts.bw_scales = [0.25 0.5 1 2 4 8];
opts.normalize_cols = true;

% stable kmeans
opts.kmeans_maxiter = 200;
opts.kmeans_replicates = 1;
opts.kmeans_start = 'plus';

opts.verbose = true;
end

function [Xz, muX, sX] = zscore_train_local(X)
muX = mean(X,1);
sX  = std(X,0,1);
sX(sX==0) = 1;
Xz = (X - muX) ./ sX;
end

function Xz = zscore_apply_local(X, muX, sX)
sX(sX==0) = 1;
Xz = (X - muX) ./ sX;
end

function K = rbf_kernel_local(A,B,gamma)
A = double(A); B = double(B);
if isvector(A), A = reshape(A, 1, []); end
AA = sum(A.^2,2);
BB = sum(B.^2,2)';
D2 = AA + BB - 2*(A*B');
D2(D2 < 0) = 0;
K  = exp(-gamma * D2);
end
