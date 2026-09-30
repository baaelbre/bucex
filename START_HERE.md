# Start here — BUCEX 1.9.8.1

Read [the screen command guide](BUCEX-1.9.8.1-commands.md).

- Gallade: `bash RUN_SWEETSPOT_HPC.sh --dry-run`, then the same command without `--dry-run`.
- BIOBOT: `bash RUN_SWEETSPOT_BIOBOT.sh --dry-run`, then the same command without `--dry-run` inside a persistent terminal session.

Set the Python environment and fresh results root first, as shown in the guide. The host batches are disjoint. The shared/private comparison keeps marginal priors matched; private fits still have shrinkage.
