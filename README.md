# ARD reconstruction reproducibility materials

Supplementary materials for **Edge Ranking and Probability Estimation from
Roster-Based Aggregated Relational Data**, by Yen-hsuan Tseng.

## Contents

`supplementary-S1.zip` contains the complete 107-file reproducibility package:
estimator and verification code, saved synthetic data and estimates, result
summaries, diagnostics, a figure, and an internal SHA-256 manifest. The study
uses forty independent graphs with 40 nodes, each observed through nested
partitions into 4 and 8 groups.

The saved execution is dated October 14, 2026. Subsequent ranking diagnostics
use those saved arrays. The archive's `README.txt` describes the fixed design,
post hoc comparisons, numerical limitations, and optional reproduction steps.
All observations are synthetic; no personal or empirical network data are used.

## Verify the saved results

1. Check the downloaded archive against `SHA256SUMS`.
2. Extract `supplementary-S1.zip` into its own directory.
3. With Python 3.9 or later and NumPy installed, run `python3 verify_all.py`
   from that directory.

This command reads and checks the saved files and results. It does not generate
new graphs, fit estimators, or write files. ReportLab is needed only to redraw
the figure. Optional full reproduction is described separately in the archive.

The public package differs from the preserved submission candidate only in one
README sentence clarifying that the optional r2 baseline archive is separate,
and in the corresponding README entry in the internal hash manifest. All
scientific files are byte-identical to the preserved candidate.
