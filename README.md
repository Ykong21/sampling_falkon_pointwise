# sampling_falkon_pointwise (MATLAB)

Pointwise (test-dependent) sampling probabilities via `kr_sampling` + pointwise retraining with FALKON.

This repo provides a **dataset-agnostic** pipeline:
for each test point `x0 = Xte(e,:)`, compute a test-dependent sampling distribution over training points using `kr_sampling`,
then resample (with replacement) and train a small FALKON model **specific to that test point**.

---

## Features
- **Pointwise sampling**: each test point has its own resampling distribution.
- **Pointwise retraining**: each test point trains its own FALKON model on the resampled subset.
- **Minimal runnable demo**: synthetic regression demo in `demo/demo_synthetic.m`.
- **Reproducible**: deterministic RNG per test point (configurable via `rng_base`).

---

## Requirements
- MATLAB (R2020b+ recommended)
- Statistics and Machine Learning Toolbox (for `kmeans`, `randsample`)
- Git (for submodules)
- A working MEX compiler (for FALKON MATLAB code)
  - On macOS: install Xcode Command Line Tools (`xcode-select --install`)
  - On Windows: Visual Studio Build Tools

### External dependency (as a submodule)
This repo includes the FALKON MATLAB implementation as a git submodule:

- https://github.com/LCSL/FALKON_paper

**Important:** The FALKON MATLAB code requires compiling two MEX files:
`inplace_chol` and `tri_solve_d`.  
This repo provides `scripts/compile_falkon_mex.m` and the demo will auto-compile when needed.

---

## Clone (with submodules)

**Option A (recommended):**
```bash
git clone --recurse-submodules https://github.com/Ykong21/sampling_falkon_pointwise.git
cd sampling_falkon_pointwise
```
## Quick start (run the synthetic demo)


