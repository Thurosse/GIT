---
name: veloline-run-inference
description: Run stage 2 (sections 9-10) of the RNA-ADT velocity pipeline (the `veloline` package): execute FIT 1 (pseudotime ϕ via SVI or NUTS) and FIT 2 (kinetics β, γ, A, B), persisting posteriors and ELBO traces to `results/<run>/state/`. Trigger ONLY when the user explicitly asks to run the velocity pipeline's inference stage — phrases like "run veloline inference stage", "run fit1", "run fit2", "execute SVI on the velocity model", "fit the Pyro models", or "rerun the variational inference". Do NOT trigger when the user is editing model definitions in `models.py`, asking conceptual questions about Pyro/SVI/MCMC, or working on plots/diagnostics.
---

# Veloline run-inference (stage 2)

Wraps notebook §9–§10. Loads the `mp` + AnnData written by `veloline-pipeline-setup`, restores RNG state, runs FIT 1 (`fit1_with_spliced_and_unspliced_model` by default) and FIT 2 (`fit2_latent_variable_model` by default), and saves the posteriors needed by stage 3.

## Prerequisites

- Stage 1 (`veloline-pipeline-setup`) has run for this dataset; the latest run dir contains `state/{adata_fit.h5ad, mp.pt, model_workflow.json, rng_state.pt}`.
- `metaparams.USE_GPU` matches the value captured by stage 1 (the inference stage aborts on mismatch — fit1 posteriors lose their device guarantees if mixed).
- `velocycle` conda env in WSL.

## Workflow

1. Run the inference stage from PowerShell:
   ```bash
   wsl -e bash -lc "cd '/mnt/c/Users/arthu/OneDrive/Bureau/Claude Code' && conda run -n velocycle python -m veloline.stages.inference"
   ```
   Optional flags:
   - `--run latest` (default) or `--run run_<UTC>_<hash>` to target a specific run dir.
   - `--stages fit1,fit2` (default) — pick a subset.
   - `--force` — re-run even if posterior files already exist (resume mode is the default).
2. The driver:
   - Restores `rng_state.pt` (calls `torch.set_default_device(device)` BEFORE any tensor allocation, then `pyro.set_rng_seed(seed)` and `torch.set_rng_state`).
   - Aborts with a clear error if `metaparams.USE_GPU` differs from the saved `use_gpu` flag.
   - Skips fit1 if `state/fit1_posteriors.pt` already exists (unless `--force`).
   - Loads fit1 PPS samples from disk to drive the fit2 constant transfer; emits a clear error if only fit2 is requested with no fit1 outputs available.
3. After completion, the new posteriors and ELBO traces are written under the run dir.

## Outputs (in `results/<run>/`)

- `state/fit1_posteriors.pt` — `{ϕ_fit, ν_fit1, ζ_fit1, ζ_dϕ_fit1?, ElogS_fit1, ElogU_fit1?, logγg_fit1?, logβg_fit1?, ElogS2_fit1, ElogU2_fit1?}`
- `state/fit1_pps.pt` — full posterior-predictive sample dict (kept so fit2 can be re-run independently).
- `state/fit2_posteriors.pt` — model-aware reconstruction with `ϕ_fit, ζ_fit, ν_fit, k_fit, βg_fit, γg_fit, Ag_fit, Bg_fit, ElogS_fit, ElogU_fit, ElogP_fit, ElogS2_fit, ElogU2_fit, ElogP2_fit`.
- `state/fit2_pps.pt` — fit2 posterior-predictive samples (consumed by §12.7 uncertainty plot in stage 3).
- `metrics/fit1_elbo.csv`, `metrics/fit2_elbo.csv` — iter,elbo CSVs (SVI only).
- `plots/fit1/elbo.png`, `plots/fit2/elbo.png`.
- `state/rng_state.pt` is re-saved after each of fit1/fit2 so a partial rerun stays deterministic.

## Failure recovery

- "USE_GPU mismatch" — toggle `USE_GPU` in `metaparams.py` to match the run, or re-run setup to mint a new run dir.
- ELBO diverges or stalls — increase `FIT_*_N_INITS` and `FIT_*_WARMUP` in `metaparams.py`; the seed search picks the lowest-loss restart.
- Out of memory on GPU — set `USE_BATCHING=True` and tune `FIT_*_CELL_BATCH_SIZE` (note: posterior predictive automatically re-runs on the full dataset).
- "fit2 requested but no fit1 posterior-predictive samples available" — run with `--stages fit1,fit2` or restore a previous `fit1_pps.pt`.
