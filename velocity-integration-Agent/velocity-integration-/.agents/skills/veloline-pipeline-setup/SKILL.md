---
name: veloline-pipeline-setup
description: Run stage 1 (sections 1-8) of the RNA-ADT velocity pipeline (the `veloline` package): load AnnData, preprocess, build the `mp` MetaparContainer, and persist `adata_fit.h5ad` + `mp.pt` + `rng_state.pt` + `model_workflow.json` to `results/<run>/state/`. Trigger ONLY when the user explicitly asks to run the velocity pipeline's setup stage — phrases like "run veloline pipeline setup", "rebuild mp", "prepare data for fit1/fit2", "execute setup stage of the velocity pipeline", or "regenerate adata_fit". Do NOT trigger on generic notebook edits, on requests to modify `metaparams.py`, on plotting questions, or on unrelated "setup" requests in other contexts.
---

# Veloline pipeline setup (stage 1)

This skill wraps notebook §1–§8 of `core_refactored_5.5.ipynb` as a single command. It mints a new run directory under `results/`, snapshots `metaparams.py`, loads the AnnData at `DATA_PATH`, runs the §6 preprocessing pipeline, builds the `mp` namedtuple via `build_mp(...)`, and writes the baseline state needed by stages 2 and 3.

## Prerequisites

- `velocycle` conda env exists with `pyro`, `scanpy`, `scvelo`, `dynamo`, `pycircstat`, `torch`, `pandas`, `scipy`, `matplotlib`, `ipywidgets`, `threadpoolctl`.
- The h5ad at `metaparams.DATA_PATH` is reachable from WSL (Linux paths inside `/home/arthur/cache/` are the convention).
- `splines_torch_fixed.py` is at `.agents/skills/veloline-pipeline-setup/scripts/` (add that path to PYTHONPATH so the package can import it).

## Workflow

1. Verify the env exists:
   ```bash
   conda env list | grep velocycle
   ```
2. Run the setup stage:
   ```bash
   conda run -n velocycle python -m veloline.stages.setup
   ```
   Optional: pass `--run-name <name>` to suffix the run directory with a memorable label.
3. After completion, run `scripts/verify_setup.py` to assert the expected files exist with the expected shapes:
   ```bash
   conda run -n velocycle python .agents/skills/veloline-pipeline-setup/scripts/verify_setup.py
   ```
4. The run dir is recorded in `results/latest.txt`; subsequent stages default to that pointer.

## Metaparameter table ritual (MANDATORY)

The editable metaparameter table is a conversational checkpoint, not just stdout.

**At init (before running setup):**
1. Run a short read-only command to dump the current editable rows:
   ```bash
   conda run -n velocycle python -c "from veloline.metaparams import get_editable_param_rows; from veloline.stages.setup import _format_editable_param_table; print(_format_editable_param_table(get_editable_param_rows()))"
   ```
2. Paste the table into chat verbatim, inside a fenced code block.
3. Ask the user: "These are the current defaults — do you want to modify any before I run setup?" using AskUserQuestion with options like {Run as-is, Modify before running}.
4. Only proceed to `python -m veloline.stages.setup` after the user confirms.

**When changing metaparameters (any edit to `veloline/metaparams.py`):**
1. Print the **before** table in chat using the same command above.
2. Make the edit with the Edit tool.
3. Print the **after** table in chat using the same command, so the diff is visible to the user.

This ritual applies whether the change is requested up front (step 3 of init) or later in the conversation.

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

- If `mp_builder` raises "No ADT-matched RNA genes are available", widen filters in `metaparams.py` (`MIN_*_MEAN`, `MIN_CELLS_FRACTION`) or set `FORCE_INCLUDE_ADT_MAPPED_GENES=True`.
- The skill **may** edit `metaparams.py` when the user asks for it during the init ritual or afterwards. Whenever it does, the before/after table ritual above is mandatory.

## Eval

A trigger-routing eval lives in [evals/](./evals/):

- [evals/cases.yml](./evals/cases.yml) — 18 labelled prompts (10 positive / 8 negative) with `expected: invoke | skip` and a rationale.
- [evals/run_eval.prompt.md](./evals/run_eval.prompt.md) — prompt file that scores the model's routing against every case and prints a pass/fail table.

Run the eval via `/run-eval-pipeline-setup` in chat. A perfect score is 18/18.
