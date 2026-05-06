---
name: veloline-pipeline-setup
description: Run stage 1 (sections 1-8) of the RNA-ADT velocity pipeline (the `veloline` package): load AnnData, preprocess, build the `mp` MetaparContainer, and persist `adata_fit.h5ad` + `mp.pt` + `rng_state.pt` + `model_workflow.json` to `results/<run>/state/`. Trigger ONLY when the user explicitly asks to run the velocity pipeline's setup stage — phrases like "run veloline pipeline setup", "rebuild mp", "prepare data for fit1/fit2", "execute setup stage of the velocity pipeline", or "regenerate adata_fit". Do NOT trigger on generic notebook edits, on requests to modify `metaparams.py`, on plotting questions, or on unrelated "setup" requests in other contexts.
---

# Veloline pipeline setup (stage 1)

This skill wraps notebook §1–§8 of `core_refactored_new.ipynb` as a single command. It mints a new run directory under `results/`, snapshots `metaparams.py`, loads the AnnData at `DATA_PATH`, runs the §6 preprocessing pipeline, builds the `mp` namedtuple via `build_mp(...)`, and writes the baseline state needed by stages 2 and 3.

## Prerequisites

- Project root: `c:\Users\arthu\OneDrive\Bureau\Claude Code` (in WSL: `/mnt/c/Users/arthu/OneDrive/Bureau/Claude Code`).
- `velocycle` conda env exists in WSL with `pyro`, `scanpy`, `scvelo`, `dynamo`, `pycircstat`, `torch`, `pandas`, `scipy`, `matplotlib`, `ipywidgets`, `threadpoolctl`.
- The h5ad at `metaparams.DATA_PATH` is reachable from WSL (Linux paths inside `/home/arthur/cache/` are the convention).
- `splines_torch_fixed.py` is at the project root (it stays there — the package imports it via PYTHONPATH).

## Workflow

1. Verify the env exists:
   ```bash
   wsl -e bash -lc 'conda env list | grep velocycle'
   ```
2. Run the setup stage from PowerShell (the working directory is set inside WSL with single-quotes to handle the space in the path):
   ```bash
   wsl -e bash -lc "cd '/mnt/c/Users/arthu/OneDrive/Bureau/Claude Code' && conda run -n velocycle python -m veloline.stages.setup"
   ```
   Optional: pass `--run-name <name>` to suffix the run directory with a memorable label.
3. After completion, run `scripts/verify_setup.py` to assert the expected files exist with the expected shapes:
   ```bash
   wsl -e bash -lc "cd '/mnt/c/Users/arthu/OneDrive/Bureau/Claude Code' && conda run -n velocycle python C:/Users/arthu/.claude/skills/veloline-pipeline-setup/scripts/verify_setup.py"
   ```
4. The run dir is recorded in `results/latest.txt`; subsequent stages default to that pointer.

## Outputs (in `results/<run>/`)

- `state/adata_fit.h5ad` — preprocessed AnnData.
- `state/adata_norm.h5ad` — log-normalised AnnData with UMAP/DPT (used by §12–§14).
- `state/data_to_fit.h5ad` — final view passed to `build_mp`.
- `state/mp.pt` + `state/mp_meta.json` — frozen MetaparContainer (schema-versioned).
- `state/model_workflow.json` — snapshot of `MODEL_WORKFLOW`.
- `state/rng_state.pt` — pyro/torch/numpy RNG state + `use_gpu` flag.
- `manifests/run_manifest.json` — pkg versions, GPU info, timing.
- `manifests/metaparams_snapshot.py` — frozen copy of `veloline/metaparams.py`.
- `logs/setup.log` — timestamped progress lines.

## Failure recovery

- If WSL paths fail, confirm the project root mounts at `/mnt/c/...`.
- If `mp_builder` raises "No ADT-matched RNA genes are available", widen filters in `metaparams.py` (`MIN_*_MEAN`, `MIN_CELLS_FRACTION`) or set `FORCE_INCLUDE_ADT_MAPPED_GENES=True`.
- The skill does not edit `metaparams.py`. If the user wants to change knobs, edit that file by hand and re-run the skill.

## Eval

A trigger-routing eval lives in [evals/](./evals/):

- [evals/cases.yml](./evals/cases.yml) — 18 labelled prompts (10 positive / 8 negative) with `expected: invoke | skip` and a rationale.
- [evals/run_eval.prompt.md](./evals/run_eval.prompt.md) — prompt file that scores the model's routing against every case and prints a pass/fail table.

Run the eval via `/run-eval-pipeline-setup` in chat. A perfect score is 18/18.
