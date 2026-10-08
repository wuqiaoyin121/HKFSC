# HKFSC MATLAB implementation

This folder contains the HKFSC optimizer, its affinity-graph construction,
the small set of functions it calls, and the frozen HKFSC parameter profiles
used in the revised manuscript. It does not include baseline implementations,
parameter-search scripts, benchmark datasets, or experimental result files.

## Requirements

- MATLAB R2021b (the version used for the manuscript experiments).
- Input matrix `X`: one finite, numeric sample per row.
- Requested number of clusters `K`: an integer from 2 through `size(X,1)-1`.

Run from this folder, or add it to the MATLAB path:

```matlab
setup_hkfsc
smoke_test
```

For one run on your own data:

```matlab
load('your_dataset.mat', 'X');
model = run_hkfsc(X, K, 'high', 1001);
labels = model.labels;
disp(model.status);
```

`run_hkfsc` sets the random seed and loads one frozen profile. The algorithm
does not read reference labels. Inspect `model.status` and `model.history`
before including a run in a summary. The manuscript's main experiment used
seeds `1001:1020`; mean and standard deviation were calculated according to
the run-inclusion rules described in the paper. This small package runs one
initialization and does not regenerate the manuscript's tables by itself.

## Frozen main-comparison profiles

| Profile | Main-comparison datasets | lambda1 | lambda2 | kNN | Initial ADMM rho | Adaptive rho | Outer iterate tolerance |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `synthetic` | Interleaved moons, circle with blob, two-armed spiral, concentric circles | 10 | 0.0001 | 10 | 1 | yes | 0.0001 |
| `low` | Aggregation, DS850, DS-577, Blobs, 3MC, Atom, Iris | 10 | 0.00001 | 15 | 0.1 | no | 0.0001 |
| `medium` | Zoo | 100 | 0.0001 | 15 | 0.1 | no | 0.0001 |
| `high` | COIL20, Yale, JAFFE, USPS | 100 | 0.01 | 15 | 0.1 | no | 0.0003 |

All profiles use `m=2`, HGK, curvature `1`, radial compression `delta=10`,
per-feature min-max normalization, and the median positive kNN-distance
bandwidth rule. They use an embedding rank equal to `K`, at most 1000 outer
iterations, at most 100 ADMM iterations and 10 membership projected-gradient
steps per outer update. The stopping rule requires the objective, membership,
embedding-subspace, and ADMM residual checks to pass for five consecutive
outer iterations after at least 20 iterations. Numeric tolerances and
backtracking settings are explicit in `hkfsc_paper_params.m`.

The profiles are transcribed from the `official_v6_simple` frozen regime
locks in the research project. The `smoke_test` deliberately uses shorter
limits and is only an installation check; it is not a paper experiment.

## Included files

- `+algorithms/+proposed/hkfsc_core.m`: optimizer and model outputs.
- `+algorithms/+proposed/default_params.m`: underlying defaults, overridden
  by the frozen profiles for manuscript runs.
- `+kernels/build_affinity.m` and `map_to_poincare.m`: graph construction.
- `+utilities/`: only the six helpers called by the above files.
- `hkfsc_paper_params.m`, `run_hkfsc.m`, `setup_hkfsc.m`, `smoke_test.m`:
  profile, entry point, path setup, and installation check.

This folder supplies the proposed algorithm and its settings. Reproducing
comparative tables additionally requires the benchmark datasets, baseline
implementations, experiment protocol, and run-level outputs.
