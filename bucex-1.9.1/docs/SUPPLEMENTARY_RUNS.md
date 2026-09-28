# Additional paper experiments

After the smoke and reference fits in [START_HERE](../START_HERE.md), use
the configured prior assessment study with `--stage plan` to inspect model
variants and resource budgets. Use `--stage all` to run full-record fits and
paired forecast checks. Seasonal `manuscript_sensitivity.json` tests the
hyperprior width and GEV shape; `physical_sensitivity.json` varies each of
the four anchors; `adequacy.json` varies the observation-scale seasonality
and latent seasonality; `dependence_sensitivity.json` varies the LKJ
concentration. Full constant-copula and independent-dependence fits have
their own configs. See [FINAL_RUN.md](../FINAL_RUN.md) for the complete
command sequence. Keep fitted outputs from each version separate.
