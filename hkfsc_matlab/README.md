# Standalone HKFSC implementation

In MATLAB R2021b, run `smoke_test` from this directory. For one run:

```matlab
model = run_hkfsc(X, K, 'high', 1001);
```

`X` has one sample per row, `K` is the requested number of clusters, and the
last argument fixes MATLAB's `twister` seed. The profiles `synthetic`, `low`,
`medium`, and `high` are defined in `hkfsc_paper_params.m`. The selected
main-dataset settings are also listed in `../configs/main_hkfsc_profiles.csv`.
`model.status` records whether the numerical stopping test was met or the
iteration limit was reached. The 20-run entry point is in `../experiments/`.
