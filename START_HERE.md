# BUCEX 1.8.9 — seasonal manuscript

Unpack on BIOBOT, install once with `python3 -m pip install -e ".[plot,test]"`,
and run `bash bash_scripts/submit_biobot.sh paper` from the release root.
This submits 106 separate four-chain analysis jobs as an eight-way PBS array,
plus five separate final fits. Use `BUCEX_MAX_JOBS=10` if ten 4-CPU/20-GB
allocations can run concurrently; set `BUCEX_PYTHON` to your environment's
absolute Python path if needed. See [FINAL_RUN.md](FINAL_RUN.md) for checks,
submission variants, collection, and figure generation.

The seasonal priors use median-absolute anchors `(0.01, 0.0001, 0.01, 0.01)`
and initial seasonal coefficient SD 20°C. The full reference fit makes 120
seasonal (30-year) predictions with 12,000 predictive draws and 95% intervals.
An integrated initial-rate prior SD near 0.96°C/decade follows from the
new initial-slope median; review its sensitivity before drawing conclusions.

The last four-chain reference took about ten hours in the previous release.
Independent jobs can overlap, but neither their completion within three hours
nor adequate convergence is assumed.
