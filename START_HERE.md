# BUCEX 1.8.8 — seasonal manuscript

From the unpacked package root, the complete paper queue is:

```bash
bash RUN_PAPER_EXPERIMENTS.sh
```

The script runs sequentially and stores logs and timestamped reports in
`results/`. See [FINAL_RUN.md](FINAL_RUN.md) to run individual experiments in
separate terminals or submit them as jobs. No scientific results in the old
1.8.7 archive are reused: the manuscript priors changed.

The four shared hyperprior medians per season are `(.01, .0001, .01, .001)`
for level innovation, slope innovation, seasonal innovation and **initial
slope**. All four are median absolute coefficients. Conditional normal SDs
divide each by `Phi^{-1}(.75)`, and the four lognormal widths are `log(2)`.
In particular, the final initial-rate prior SD is approximately 0.096°C per
decade after integrating the hyperprior. Preflight reports the conversion.

The full-record fit has 538 seasons from MAM 1892 through JJA 2026 and uses
four chains with 3,000 warm-up and 8,000 retained draws each. Convergence
gates control which run reports can be interpreted or plotted.
