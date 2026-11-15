# Count Constraints and Conditional Benchmarks for Roster-Based Network Reconstruction

**Replication materials.** All networks and observations are simulated. The archive includes the original inputs, fitted probability summaries, conditional marginals and numerical code needed to reproduce each manuscript table and figure. The short route starts from the archived fitted matrices. The separate generation and refitting commands start from the recorded random seeds.

## Requirements

The scientific programs use Python 3.12 and NumPy 2.3.5. For figure PDFs install a TeX distribution providing pdfLaTeX, standalone, pgfplots and Latin Modern. Table and figure data generation uses Python alone. Run the following from the directory containing this README, after installing NumPy into your interpreter of choice.

```sh
python -m pip install -r requirements.txt
python -B reproduce.py verify
python -B reproduce.py rescore --out ../paper-results --compile-figures
```

The rescore command also runs the exploratory saved-data check and rebuilds all four exploratory tables. Use a new output directory for each run. Omit `--compile-figures` for the same numerical checks and the editable TeX figure sources without PDF compilation.

| Paper item | Data and computation | Output under the chosen destination |
|---|---|---|
| Table 1 | Fresh circular blocks, zero exclusions, constrained pairs and forced-state AUC | `density-design.tex` |
| Table 2 | Unresolved-pair AUC and within-block differences | `density-ranking.tex` |
| Table 3 | Probability RMSE, relative reductions and paired ranking differences | `density-estimation.tex` |
| Table 4 | Circular 15% and SBM constraint frequencies, within-SBM paired differences | `sbm-context.tex` |
| Table 5 | SBM ranking, probability RMSE and excess Brier risk | `sbm-comparison.tex` |
| Table 6 and Appendix Tables A1–A3 (tables 8–10 in sequence) | Exploratory circular sample, final iterates and four-chain means | `exploratory-tables/` and its `table-validation.json` |
| Table 7 | Corrected fitted-weight marginal probabilities on all five cohorts, including the exploratory cohort | `plugin-comparison.tex` |
| Figure 1 | Exhaustive enumeration at row and column margins (3,3,1,1), yielding four configurations | `positive-margins.tex`, configurations JSON, optional PDF |
| Figure 2 | Seven mean AUCs and one-MCSE bars at three densities, using the full-precision graph-level results | `density-unresolved-auc.tex`, data JSON, optional PDF |

## Data and estimands

- `studies/fresh/` contains forty independent seed blocks 21001–21040. Each block reuses its angles and edge uniforms at three densities and both partitions. This yields 120 realized graphs and 240 graph–partition cases.
- `studies/sbm/` contains forty independent graphs, seeds 31001–31040, and one eight-group observation partition. The latent four-community assignments and the edge uniforms are in `inputs/blocks/`. Roster membership and generating community membership are separate variables.
- Each graph input contains observed counts `Y`, partition matrix `B`, realized adjacency `hidden_A` and generating probabilities `true_P`. The fit uses `Y,B`. Evaluation uses the realized graph and generating probabilities for their respective targets. The conditional oracle also uses the generating odds.
- `circular/*/run_*/summary.npz` contains the pooled, chain-specific, half-chain and chain-group mean probability matrices. `diagnostics.npz` contains entrywise and scalar diagnostics. Nuclear archives contain early and final feasible probability matrices for all three fixed penalties. Per-fit JSON records retain stopping status and residuals. All prescribed fits, including capped iterates, contribute.
- Conditional archives at eight groups contain exact integer support counts, forced and unresolved masks, uniform marginals and known-generating-law oracle marginals. The plug-in archives include direct fits, their clipped comparators, weighted marginals and diagnostic log marginal odds for 200 cases. Each case has one circular and nine nuclear plug-in matrices from the three penalties and three clipping thresholds.
- `reference/` includes the manuscript table bodies, the 21 full-precision Figure 2 coordinates, original aggregate CSVs, graph-level clipping/conditioning diagnostics, and selected records from the original numerical checks. The primary AUC gives half credit for exact ties in stored float64 conditional probabilities. The log-odds rankings have their own diagnostic fields.
- The manuscript's exploratory S1 source is fixed at commit `9fa2b75305216b279cb03501c0d7a56b653a3ae8`. Its graph and early-estimator lineage is distinct from the extended exploratory fits and the independent fresh samples. The local reproduction commands operate offline after dependencies are installed. [Public S1 release at the fixed commit](https://github.com/kwt-klure/ard-reconstruction-reproducibility/tree/9fa2b75305216b279cb03501c0d7a56b653a3ae8)

## Recreate the conditional references and plug-in analysis

```sh
python -B code/check_references.py --output ../independent-references.json --workers 4
python -B reproduce.py reweight --out ../reweighted --workers 4
python -B reproduce.py rescore --matrices ../reweighted --out ../paper-from-new-weights
```

The first command uses separate edge-fixed dynamic programming and within-group enumeration to check the saved support counts, masks and both reference probabilities. The second applies the fitted odds to the exact supports and checks every resulting array against its archived counterpart. Its case JSON includes the original diagnostic rankings. The third command applies the corrected probability-based AUC readout and rebuilds the manuscript results. Each cohort contains forty graph replicates. The three fresh circular panels share seed blocks, and contrasts are formed within graphs before computing the MCSE.

## Regenerate the fresh circular and SBM studies

```sh
for study in fresh sbm
do
  python -B code/regenerate.py "$study" generate "../rerun-$study" --compare-inputs
  python -B code/regenerate.py "$study" conditional "../rerun-$study" --workers 2
  python -B code/regenerate.py "$study" nuclear "../rerun-$study" --workers 4
  python -B code/regenerate.py "$study" circular "../rerun-$study" --workers 4
done
python -B reproduce.py reweight --fresh ../rerun-fresh --sbm ../rerun-sbm --out ../refitted-weights
python -B reproduce.py rescore --fresh ../rerun-fresh --sbm ../rerun-sbm --matrices ../refitted-weights --out ../refitted-results
```

The generation step checks every synthetic array against the deposit. The three subsequent phases write original-format results into the new directories. For the exploratory sample, use the regeneration instructions under `exploratory/`. To use a regenerated exploratory tree in the plug-in commands, pass `--exploratory PATH` pointing to the directory containing `jspi-rerun-2026-10-14`, `compstat-stability-2026-11-04` and `compstat-conditional-2026-11-04`.

The four-chain sampler retains 3,200 draws per chain after 1,600 warmup sweeps. Its raw dyad-probability arrays occupy approximately 19.2 GB for fresh circular fits, 3.2 GB for SBM fits and 6.4 GB for exploratory fits. Allow additional space for regeneration outputs and ample CPU time. The compact deposit includes the fit summaries and generation code. Recomputing draw-level posterior means or diagnostics requires regeneration of those traces. A different numerical library, BLAS implementation or long-double precision may affect floating-point ties and refitting trajectories. The archived-array route fixes the fitted matrices used for the paper.

## Licenses

Yen-hsuan Tseng licenses the scientific Python code under MIT and the original synthetic data, simulation outputs, documentation, and table and figure materials under CC BY 4.0. Third-party dependencies retain their own licenses. [Archive license and scope](LICENSE.md) · [MIT](https://opensource.org/license/mit) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
