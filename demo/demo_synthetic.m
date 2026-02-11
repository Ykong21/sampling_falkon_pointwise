function demo_synthetic()
% demo_synthetic
% Minimal runnable demo for sampling_falkon_pointwise.
% Generates synthetic regression data, runs pointwise sampling+FALKON pipeline,
% and reports basic metrics + simple plots.

clc; rng(1025,'twister');

repo_root = fileparts(fileparts(mfilename('fullpath')));

% --- Paths ---
addpath(fullfile(repo_root,'src'));
addpath(fullfile(repo_root,'scripts'));
addpath(genpath(fullfile(repo_root,'external','FALKON_paper')));

% --- Ensure mex is available ---
compile_falkon_mex();  

%% ---- synthetic data ----
Ntrain = 20000;
test_n = 500;
p = 30;

Xtr = randn(Ntrain,p);
f = @(X) ...
    2*sin(X(:,1)) + ...                 
    0.5*(X(:,2).^2) + ...             
    1.5*tanh(X(:,3)) + ...              
    0.8*(X(:,4).*X(:,5)) + ...          
    1.2*exp(-0.5*(X(:,6).^2));          

sigma_eps = 0.5;
ytr = f(Xtr) + sigma_eps*randn(Ntrain,1);


Xte = randn(test_n,p);
yte = f(Xte) + sigma_eps*randn(test_n,1);

%% ---- run pipeline ----
out = sampling_falkon_pointwise(Xtr, ytr, Xte, ...
    'n_sub', 5000, ...
    'M_centers', 2000, ...
    'falkon_iters', 10, ...
    'memToUseGB', 4, ...
    'useGPU', 0, ...
    'verbose', true);

yhat = out.yhat;

mse = mean((yte - yhat).^2);
relL2 = norm(yhat - yte) / norm(yte);

fprintf('\n=== Synthetic demo results ===\n');
fprintf('MSE   = %.6f\n', mse);
fprintf('relL2 = %.6f\n', relL2);
fprintf('Select time (kr_sampling) = %.3fs\n', out.select_time_s);
fprintf('Train total (FALKON)      = %.3fs\n', sum(out.train_time_s));

%% ---- plots ----
err = yhat - yte;

figure;
plot(yte, yhat, '.'); grid on;
xlabel('y true'); ylabel('y hat');
title('Pointwise sampling+FALKON: prediction scatter');

figure;
histogram(err, 30);
grid on;
xlabel('prediction error'); ylabel('count');
title('Prediction error histogram');

end
