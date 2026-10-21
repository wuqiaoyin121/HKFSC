# Standalone HKFSC

From this directory in MATLAB R2021b, run `smoke_test`. It uses a small generated
matrix and reduced iteration limits; a finite `max_iter` output passes this
call-chain/finite-value test without establishing convergence.

```matlab
model = run_hkfsc(X, K, 'high', 1001);
```

X has samples in rows; K is prespecified. Reference assignments are not inputs
to the fit. The last argument fixes MATLAB's twister seed. Paper profiles
`synthetic`, `low`, `medium`, `high` are in `hkfsc_paper_params.m`.
`model.status` distinguishes convergence from an iteration limit.

For the recorded main-comparison settings across HKFSC and baselines, return to
the repository root and call `setup_hkfsc_project`, then
`run_main_comparison('iris',20)` or `run_main_hkfsc_20('iris')`.
See `../LICENSE` for the authors' all-rights-reserved decision.
