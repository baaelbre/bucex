# BUCEX 1.9.3

[HPC commands and complete grid](BUCEX-1.9.3-commands.md) · [Verification](validation/RELEASE_VALIDATION_193.md)

The active seasonal workflow now uses six independent analyses with fixed Normal shrinkage priors. The reference median absolute coefficients are `(0.01, 0.0001, 0.01, 0.01)`, including the initial slope; the fixed Normal SD is each value divided by `Phi^-1(.75)`. All private and shared shrinkage hyperpriors are removed from this grid. General library support for historical hierarchical/copula fits remains available.

Sensitivity acts directly on the four Normal prior SDs. The 23 full-record settings include individual half/double controls, global half/double controls, quarter seasonal SD, doubled-slope/stronger-seasonal combinations, shape priors and observation/seasonality alternatives. All six responses run for each setting; Gaussian means under shape-only changes are unchanged controls.

HPC arrays now allocate one experiment per element, with six simultaneous response fits and two chains each: twelve cores for both screen and paper. Paper increases iterations and tightens diagnostics. A fresh compute-node environment setup, standard-library-only login submitter, mandatory compute probe, physical executable paths, response logs, provenance checks and preserved failures address the earlier startup failure modes. Result collection can also run on a compute node.

`RUN_*_EXPERIMENTS.sh` submits 27 jobs/142 fits. `RUN_*_ALL.sh` adds both validation designs, giving 88 jobs/508 fits. Keep 1.9.3 in a new results root. No production fit or live HPC submission has been performed in this workspace.
