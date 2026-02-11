function compile_falkon_mex(varargin)
% compile_falkon_mex
% Compile required FALKON mex files for MATLAB implementation in this repo.
%
% Default behavior:
%   - If mex binaries already exist, do nothing.
%   - Otherwise compile:
%       inplace_chol.cpp
%       tri_solve_d.cpp
%   - Preferred link flags on mac/linux/windows:
%       -lmwlapack -lmwblas   (MATLAB-provided LAPACK/BLAS)
%
% Usage:
%   compile_falkon_mex()
%   compile_falkon_mex('force', true)   % recompile even if mex exists

p = inputParser;
p.addParameter('force', false, @(x) islogical(x) || isnumeric(x));
p.addParameter('verbose', true, @(x) islogical(x) || isnumeric(x));
p.parse(varargin{:});
opt = p.Results;

repo_root = fileparts(fileparts(mfilename('fullpath')));
fdir = fullfile(repo_root, 'external', 'FALKON_paper', 'FALKON');

if ~isfolder(fdir)
    error('[compile] FALKON dir not found: %s\nDid you clone with --recurse-submodules?', fdir);
end

% mex output extension (platform-specific)
mexext_str = mexext;

mex_inplace = fullfile(fdir, ['inplace_chol.' mexext_str]);
mex_tri     = fullfile(fdir, ['tri_solve_d.' mexext_str]);

if ~opt.force && isfile(mex_inplace) && isfile(mex_tri)
    if opt.verbose
        fprintf('[compile] FALKON mex already present:\n  %s\n  %s\n', mex_inplace, mex_tri);
    end
    return;
end

if opt.verbose
    fprintf('[compile] Using FALKON dir: %s\n', fdir);
end

% Ensure we compile into the same folder as the sources
cwd0 = pwd;
cleanup = onCleanup(@() cd(cwd0));
cd(fdir);

try
    % Preferred: link MATLAB's LAPACK/BLAS
    if opt.verbose, fprintf('[compile] mex inplace_chol.cpp (link: -lmwlapack -lmwblas)\n'); end
    mex('-O','-largeArrayDims', fullfile(fdir,'inplace_chol.cpp'), '-lmwlapack', '-lmwblas');

    if opt.verbose, fprintf('[compile] mex tri_solve_d.cpp (link: -lmwlapack -lmwblas)\n'); end
    mex('-O','-largeArrayDims', fullfile(fdir,'tri_solve_d.cpp'),  '-lmwlapack', '-lmwblas');

    if opt.verbose
        fprintf('[compile] Success.\n');
        fprintf('[compile] Built:\n  %s\n  %s\n', mex_inplace, mex_tri);
    end

catch ME
    fprintf(2,'[compile] MEX build failed.\n');
    fprintf(2,'[compile] Tip: run `mex -v ...` to see full link line.\n');
    rethrow(ME);
end

end
