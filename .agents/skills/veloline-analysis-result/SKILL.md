---
name: veloline-analysis-result
description: Run stage 3 (sections 11-14) of the RNA-ADT velocity pipeline (the `veloline` package): generate fit-quality plots, velocity field, RNA-vs-ADT shift analysis, and lymphoid/myeloid co-expression plots from saved posteriors in `results/<run>/state/`, persisting figures to `results/<run>/plots/` and metrics to `results/<run>/metrics/`. Trigger ONLY when the user explicitly asks to produce velocity-pipeline analysis figures — phrases like "run veloline analysis stage", "regenerate velocity plots", "run shift analysis", "produce fit-quality figures", or "make co-expression plots from latest run". Do NOT trigger on ad-hoc plot tweaks inside the notebook, requests for matplotlib styling help, or new analytical experiments unrelated to the pipeline.
---

# Veloline analysis-result (stage 3)

Wraps notebook §11–§14. Loads the saved posteriors from a previous inference run and generates four sets of figures (with paired numeric metrics):

- **§11 fit-quality** — phi-vs-counts overlays, P_fits, S/U fits.
- **§12 velocity field** — UMAPs coloured by velocity/drift magnitude, RNA & ADT correlation scatters, gene-cell alignment, per-cell uncertainty colouring.
- **§13 shift analysis** — windowed dS/dϕ vs dP/dϕ, bar plot of divergence scores, per-gene panels for the top-N most divergent genes.
- **§14 co-expression** — lymphoid×myeloid threshold sweep, UMAP overlay of the most-common ADT×RNA pair, KDE / raw-count histograms for that population.

## Prerequisites

- Stage 2 (`veloline-run-inference`) has populated `state/{adata_norm.h5ad, data_to_fit.h5ad, mp.pt, fit1_posteriors.pt, fit2_posteriors.pt, rng_state.pt}`. Section 12.7 also requires `state/fit2_pps.pt`.
- `velocycle` conda env in WSL.

## Workflow

1. Run the analysis stage:
   ```bash
   conda run -n velocycle python -m veloline.stages.analysis
   ```
   Optional flags:
   - `--run latest` (default) or `--run run_<UTC>_<hash>` to target a specific run.
   - `--sections fit_quality,velocity,shift,coexpression` (default) — pick any subset.

## Outputs (in `results/<run>/`)

```
plots/
├── fit_quality/   phi_vs_counts.png, P_fits_*.png, SU_fits_*.png
├── velocity/      rna_velocity_drift_scatter.png, rna_velocity_drift_uncertainty.png,
│                  rna_velocity_alignment.png, adt_velocity_drift_scatter.png
├── shift/         shift_barplot.png, shift_gene_<NAME>.png (top-N)
└── coexpression/  coexpr_top_pair_umap.png, coexpr_kde_model_expectations.png,
                   coexpr_kde_raw_counts.png, coexpr_raw_count_hist.png
metrics/
├── velocity_alignment.json   {rna_pearson, rna_spearman, adt_pearson, adt_spearman}
├── shift_scores.csv          gene, mean_|dS/dϕ - dP/dϕ| (sorted descending)
└── coexpression_top20.csv    threshold, rank, lymphoid_adt, myeloid_rna, count
logs/analysis.log
```

## Failure recovery

- "USE_GPU mismatch" — same root cause as inference stage; re-run setup or restore the original flag.
- "Could not align gene names to matrix rows" — `mp.fit2_gene_names` was overwritten unexpectedly; rebuild `mp` by re-running stage 1.
- §14 co-expression raises "No marker genes found" — the gene set in `LYMPHOID_MARKERS` / `MYELOID_MARKERS` (defined in `veloline/analysis/coexpression.py`) doesn't intersect with `mp.fit2_gene_names`. Either widen the marker lists or relax the gene filters in `metaparams.py` and re-run setup.

## Eval

A trigger-routing eval lives in [evals/](./evals/):

- [evals/cases.yml](./evals/cases.yml) — 18 labelled prompts (10 positive / 8 negative) with `expected: invoke | skip` and a rationale.
- [evals/run_eval.prompt.md](./evals/run_eval.prompt.md) — prompt file that scores the model's routing against every case and prints a pass/fail table.

Run the eval via `/run-eval-analysis-result` in chat. A perfect score is 18/18.
