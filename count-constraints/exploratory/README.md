# Exploratory sample and numerical stability

The same 40 simulated graphs were observed under partitions of four and eight groups. The archive contains the 80 observed count arrays and underlying graphs, original fitted matrices, 80 longer circular fits, and 240 nuclear fits at penalties 0.25, 1 and 4. Exact uniform and known-probability conditional references are provided for the eight-group cases. The independent density and blockmodel studies use the top-level `studies/` directory.

## Check the saved results

From the root of the repository, run the following in a Python environment with NumPy installed. The optional report argument names a new file outside the archive.

```sh
python -B exploratory/verify_saved.py --report ../exploratory-verification.json
```

The verifier checks file membership and SHA-256 hashes, the identity between adjacency matrices and observed counts, independent conditional probabilities and support counts for all 40 graphs, solver checkpoints, inclusion of all five iteration-capped nuclear fits, circular chain means, scalar traces, and diagnostic summaries. It recomputes the exploratory score summaries and paired comparisons from the saved matrices in a temporary directory. The tie-sensitivity calculation uses all eight-group graphs. The manuscript table comparison is provided separately in the repository.

## Recompute from the inputs

```sh
python -B exploratory/reproduce.py --workspace ../exploratory-new --lane conditional
```

Each run creates a new directory. `--lane conditional` enumerates the eight-group compatible graphs block by block and checks each conditional result with a separate implementation. `--lane circular` and `--lane nuclear` run the respective fitting procedures. `--lane all` runs all three, rebuilds the stability summaries and checks the regenerated outputs. `--prepare-only` copies and checks the required inputs without running the analyses. The original environment used Python 3.12 and NumPy 2.3.5.

The 80 per-draw probability files occupy 6,389,770,240 bytes in the exploratory sample and are omitted. The output of a full rerun includes these files and requires additional space. Saved chain means, scalar traces and diagnostic arrays support the checks above. Recomputing every probability-coordinate diagnostic from its original draws requires regenerating the full traces.

The directories under `project/analysis/ard-reconstruction/` retain their original relative layout. `jspi-rerun-2026-10-14/fresh-results/` supplies the original matched graph inputs and fits, `compstat-conditional-2026-11-04/` supplies the eight-group references, and `compstat-stability-2026-11-04/` contains the longer fits and penalty study. The `fresh-results` name refers to the original exploratory computation. The two later studies use separate samples.

`MANIFEST.json` describes this distribution. `ORIGINAL_RUN_HASHES.json` identifies the original method/specification records referenced inside the saved files. This distribution omits local run administration and uses portable specifications. The archived `.npy` and `.npz` files are byte-identical to the source release candidate. Numerical settings and the case set are unchanged.
