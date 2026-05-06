# AGENTS.md

This file provides guidance to AI Agent (copilot, claude ,codex,...) when working with code in this repository.

## Project

A probabilistic **RNA–ADT velocity pipeline**: a variational Bayes (Pyro) workflow that infers RNA-velocity-like dynamics from paired RNA + protein (ADT) single-cell data.

The codebase exists in **two synchronised forms**:

- [core_refactored_new.ipynb](core_refactored_new.ipynb) — the original linear notebook driver (87 cells, 14 numbered sections).
- [veloline/](veloline/) — the same pipeline refactored into an importable Python package, organised into three runnable stages (setup → inference → analysis). This is the canonical form for reproducible runs.
- [veloline_driver.ipynb](veloline_driver.ipynb) — a thin 31-cell driver notebook that imports from `veloline` and calls its public API; use this instead of the original notebook when running the package CLI interactively.

[splines_torch_fixed.py](splines_torch_fixed.py) is the device-aware torch B-spline basis module, imported by both forms.

There is no test suite and no build step.

## Environment setup

```bash
# Install dependencies (CPU-only torch)
pip install -r requirements.txt

# For CUDA (replace cu121 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Register the velocycle kernel for Jupyter (if using notebooks)
python -m ipykernel install --user --name velocycle --display-name "velocycle"

# Launch Jupyter
jupyter notebook
```

The notebooks expect the kernel named **`velocycle`**. `USE_GPU` in [veloline/metaparams.py](veloline/metaparams.py) drives `torch.set_default_device(device)` at import time — set it before running any stage.

## Running the pipeline

### Package CLI (canonical)

Three stages, each consuming the artefacts of the previous one. Each stage reads `veloline/metaparams.py` directly.

```bash
python -m veloline.stages.setup       [--run-name NAME]
python -m veloline.stages.inference   [--run latest|NAME] [--stages fit1,fit2] [--force]
python -m veloline.stages.analysis    [--run latest|NAME] [--sections fit_quality,velocity,shift,coexpression]
```

Run a single analysis section only:
```bash
python -m veloline.stages.analysis --run latest --sections velocity
```

Re-run inference from scratch on an existing setup (overwrite posteriors):
```bash
python -m veloline.stages.inference --run latest --force
```

### Notebooks

- [veloline_driver.ipynb](veloline_driver.ipynb) — preferred; delegates to `veloline` package, ~31 cells. Contains two commented **resume cells** (§8 and §10) to reload state after a kernel restart without re-fitting.
- [core_refactored_new.ipynb](core_refactored_new.ipynb) — original; all logic is inline. Open with the `velocycle` kernel.

For both: data paths in `data_all_path` are Linux paths (`/home/arthur/cache/...`). On Windows the workflow runs via WSL or a remote Jupyter kernel — do not convert paths to Windows form unless asked.

## Agent automation

This repo ships skill files that wrap the three pipeline stages into repeatable commands. Use these instead of re-deriving run steps in chat:

- [veloline-pipeline-setup](.agents/skills/veloline-pipeline-setup/SKILL.md) — stage 1 (setup)
- [veloline-run-inference](.agents/skills/veloline-run-inference/SKILL.md) — stage 2 (fit1/fit2)
- [veloline-analysis-result](.agents/skills/veloline-analysis-result/SKILL.md) — stage 3 (analysis)

## Notebook ↔ package mapping

The 14 numbered notebook sections map 1:1 onto the package:

| § | Notebook section | Package module |
|---|------------------|----------------|
| 1, 2 | Imports + Meta-parameters | [veloline/metaparams.py](veloline/metaparams.py) |
| 3 | Utility & fitting functions | [veloline/utils.py](veloline/utils.py) |
| 4 | Visualisation functions | [veloline/viz.py](veloline/viz.py) |
| 5 | Data loading | [veloline/data_loading.py](veloline/data_loading.py) |
| 6 | Pre-processing | [veloline/preprocess.py](veloline/preprocess.py) |
| 7 | Probabilistic models | [veloline/models.py](veloline/models.py) |
| 8 | Build `mp` container | [veloline/mp_builder.py](veloline/mp_builder.py) |
| 9 | FIT 1 | [veloline/fit1.py](veloline/fit1.py) |
| 10 | FIT 2 | [veloline/fit2.py](veloline/fit2.py) |
| 11 | Fit-quality plots | [veloline/analysis/fit_quality.py](veloline/analysis/fit_quality.py) |
| 12 | Velocity field | [veloline/analysis/velocity.py](veloline/analysis/velocity.py) |
| 13 | Shift analysis | [veloline/analysis/shift.py](veloline/analysis/shift.py) |
| 14 | Co-expression population analysis | [veloline/analysis/coexpression.py](veloline/analysis/coexpression.py) |

The `veloline/__init__.py` docstring is the authoritative section→module map; keep it in sync if a section moves. When a change touches both forms, edit the package module first and mirror to the notebook — the package is the canonical source.

## Run directory layout

`veloline/stages/setup.py` mints a fresh run directory under `results/` and writes `latest.txt` so subsequent stages can resolve `--run latest`. Per-run layout:

```
results/
├── latest.txt
└── run_<UTC-timestamp>_<short-hash-or-name>/
    ├── state/           # adata_fit.h5ad, adata_norm.h5ad, data_to_fit.h5ad,
    │                    # mp.pt, mp_meta.json, model_workflow.json, rng_state.pt,
    │                    # fit1_posteriors.pt, fit1_pps.pt, fit2_posteriors.pt, fit2_pps.pt
    ├── plots/           # fit1/, fit2/ (ELBO), fit_quality/, velocity/, shift/, coexpression/
    ├── metrics/         # CSV exports (e.g. fit1_elbo.csv)
    ├── logs/            # one .log per stage
    └── manifests/       # run_manifest.json, metaparams_snapshot.py
```

State save/load is centralised in [veloline/io_state.py](veloline/io_state.py). `mp.pt` is gated by `MP_SCHEMA_VERSION` — bumping the `MetaparContainer` namedtuple shape requires bumping that constant, otherwise `load_mp` will refuse stale runs.

## The architectural rule

**All tunable knobs live in [veloline/metaparams.py](veloline/metaparams.py).** The notebook reads them via `from veloline.metaparams import *`; the stage CLIs read them directly. Never scatter hyperparameters into other modules. The derived `device` and `torch.set_default_device(device)` call at the bottom of `metaparams.py` are intentionally below the **"do NOT edit below this line"** banner.

Stage gates — `MODEL_WORKFLOW["fit_1"]["enabled"]` / `["fit_2"]["enabled"]` — skip the heavy SVI/MCMC stages. Preserve those guards when editing.

## Two-stage generative hierarchy

- **FIT 1** infers per-cell pseudotime ϕ from RNA counts. Two variants:
  - `fit1_latent_variable_model` — spliced S only.
  - `fit1_with_spliced_and_unspliced_model` — joint S + U with kinetic rate priors β, γ.
- **FIT 2** conditions on ϕ (transferred via `FIT_1_TO_FIT_2_CONSTANTS`) and infers kinetic rates β, γ (RNA) and A, B (ADT). Four interchangeable variants:
  - `fit2_latent_variable_model` — dual derivative splines (separate RNA / ADT knot grids).
  - `fit2_latent_variable_model0` — legacy single shared derivative spline.
  - `fit2_latent_integral_model` — analytic integral of the RNA spline drives protein P (4-term Taylor expansion using spline derivatives up to 3rd order).
  - `fit2_latent_integral_separate_model` — same integral but only P is observed (ADT-focused diagnostic).

All six are registered in `MODEL_REGISTRY` (top of [veloline/models.py](veloline/models.py)). **Adding a new variant requires both writing the function and adding it to `MODEL_REGISTRY`** — the registry is what `MODEL_WORKFLOW`'s string names resolve against.

## State carriers

- **`mp`** — `MetaparContainer` namedtuple built in [veloline/mp_builder.py](veloline/mp_builder.py). Passed verbatim to every Pyro model and helper. Lifecycle: gene-index resolution → ADT matching → tensor construction (S, U, P + size factors) → fit1/fit2 constant injection.
- **`MODEL_WORKFLOW`** — single dict consumed by `_run_backend`. Has `fit_1` and `fit_2` keys with `enabled`, `model_name`, `inference_backend`, `observables`, `mcmc`, etc. Snapshotted to `state/model_workflow.json` per run.
- **`FIT_1_TO_FIT_2_CONSTANTS`** — list of fit1 posterior summaries frozen into `mp` for fit2. Default: `["phi","nu","logbeta","loggamma","shape_inv"]`. **Do not include `spliced` or `unspliced`** — they are observation tensors, not latents to transfer.

## The torch B-spline module

[splines_torch_fixed.py](splines_torch_fixed.py) provides device-aware B-spline basis evaluation, imported by both `veloline/models.py` and the notebooks. Two recursion implementations exist: `torch_B` (faster, mask-indexed) and `tvect_B` (fully vectorial via `torch.where`, kept for reference). Prefer `torch_B` for production use.

## Conventions to respect when editing

- Greek-letter Python identifiers (`ϕ`, `ζ`, `β`, `γ`, `ν`, `σ`) are intentional — do not ASCII-ify them.
- Section banners use box-drawing characters (e.g. `# ── 6.4 Build model-fitting object ──────────`). Preserve that style.
- The pipeline expects `adata` to have: `.X` (spliced), `.layers["unspliced"]`, `.obsm["X_adt"]`, `.obsm["X_umap"]`, `.obs[cell_id]` (cluster column), `.uns["adt_var_names"]`. Do not invent alternative slot names.
- The ADT → RNA gene mapping (`adt_rna_map`) is a hand-curated dictionary; many entries are intentionally `None`. Edit it carefully and only when adding new ADT proteins.
