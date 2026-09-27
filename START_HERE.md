# BUCEX 1.9.0

The complete commands, settings and result audit are in [FINAL_RUN.md](FINAL_RUN.md).
In the unpacked release directory, use your existing VSC environment:

```bash
export BUCEX_PYTHON="$(command -v python3)"
"$BUCEX_PYTHON" -m pip install -e '.[plot,test]'

# Shorter screen: 25 posterior fits, 2 chains per fit.
bash RUN_SCREEN_EXPERIMENTS.sh

# Separate full paper runs: 25 posterior fits, 4 chains per fit.
bash RUN_PAPER_EXPERIMENTS.sh
```

Use one tier at a time as appropriate. These commands do not submit validation
or pre-2019 jobs. Add `--dry-run` to inspect a submission without launching it.
Both tiers save fit archives and 30-year forecasts with 95% intervals.
The reference has log(3) hyperprior width, with second-moment-matched anchors;
24 alternatives examine widths, individual anchors, shape, scale and dependence.
Paper fits require numerical review before their results enter the manuscript.
