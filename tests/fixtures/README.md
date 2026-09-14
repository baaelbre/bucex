# Historical result fixtures

`v1_5_2_gaussian.bucex` and `v1_5_2_gev.bucex` were generated using the original
uploaded, unchanged BUCEX 1.5.2 source, not by relabelling current archives.
They contain synthetic data only and archive schema 2.7.0.

Generation used 12 `numpy.random.default_rng(seed)` normal/Gumbel observations,
respectively; Gaussian seed 16201, GEV seed 16202. Fits used `period=4`,
`priors="normal"`, `parameterization="fs"`, three retained draws, two warmup
iterations and one chain with fitting seed `seed+10`. Engines were `ffbs` and
explicitly approximate `laplace`. These tiny fits test archive compatibility;
they are not inference-validation examples.
