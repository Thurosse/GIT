"""Stage 1 — pipeline setup (notebook §1–8).

Run:
    python -m veloline.stages.setup [--run-name NAME]

Reads `metaparams.py`, loads the AnnData, preprocesses, builds `mp`, and writes
all baseline state to `results/<run>/state/` plus a manifest snapshot of
`metaparams.py`.
"""

import argparse
import os
import time

from veloline.metaparams import (
    DATA_PATH, BARCODE_NAMES, BARCODE_SELECTED, USE_GPU, MODEL_WORKFLOW,
)
from veloline.data_loading import load_adata, demultiplex_barcode
from veloline.preprocess import preprocess_pipeline
from veloline.mp_builder import build_mp
from veloline import io_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None,
                    help="Optional human-readable suffix for the run dir; default uses a short hash.")
    args = ap.parse_args()

    t0 = time.time()
    run_dir = io_state.mint_run_dir(name=args.run_name)
    print(f"[setup] run_dir = {run_dir}")
    io_state.append_log(run_dir, "setup", f"run dir created: {run_dir}")

    io_state.write_manifest(run_dir, extra={"data_path": DATA_PATH, "use_gpu": USE_GPU})
    io_state.freeze_metaparams(run_dir)

    print(f"[setup] loading {DATA_PATH}")
    adata = load_adata(DATA_PATH)
    adata = demultiplex_barcode(adata, BARCODE_NAMES, BARCODE_SELECTED)

    print("[setup] preprocessing (filter → HVG → adata_fit → size factors → data_to_fit)")
    adata_fit, adata_norm, data_to_fit = preprocess_pipeline(adata)

    io_state.save_adata(adata_fit, os.path.join(run_dir, "state", "adata_fit.h5ad"))
    io_state.save_adata(adata_norm, os.path.join(run_dir, "state", "adata_norm.h5ad"))
    io_state.save_adata(data_to_fit, os.path.join(run_dir, "state", "data_to_fit.h5ad"))

    print("[setup] building mp")
    mp = build_mp(adata, data_to_fit)
    io_state.save_mp(mp, run_dir)

    io_state.save_workflow(MODEL_WORKFLOW, run_dir)
    seed = int(os.environ.get("VELOLINE_SEED", 0))
    io_state.save_rng_state(run_dir, seed=seed, use_gpu=USE_GPU)

    elapsed = time.time() - t0
    io_state.append_log(run_dir, "setup", f"done in {elapsed:.1f}s")
    io_state.write_manifest(run_dir, extra={
        "data_path": DATA_PATH, "use_gpu": USE_GPU,
        "setup_seconds": round(elapsed, 1),
        "n_cells": data_to_fit.n_obs, "n_genes_fit1": int(mp.Ng_fit1), "n_genes_fit2": int(mp.Ng_fit2),
    })
    print(f"[setup] complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
