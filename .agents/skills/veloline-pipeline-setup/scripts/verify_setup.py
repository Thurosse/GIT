"""Sanity-check the setup stage's outputs in the latest run dir."""

import os
import sys

from veloline import io_state


def main():
    run_dir = io_state.latest_run_dir()
    state = os.path.join(run_dir, "state")
    required = ["adata_fit.h5ad", "adata_norm.h5ad", "data_to_fit.h5ad",
                "mp.pt", "mp_meta.json", "model_workflow.json", "rng_state.pt"]
    missing = [f for f in required if not os.path.exists(os.path.join(state, f))]
    if missing:
        print(f"[verify_setup] MISSING: {missing}")
        sys.exit(1)

    mp = io_state.load_mp(run_dir)
    expected_fields = {"S", "U", "P", "Ng", "Nc", "S_fit1", "U_fit1", "S_fit2", "U_fit2", "P_fit2",
                       "fit1_gene_names", "fit2_gene_names", "spline_t", "spline_k", "device"}
    actual = set(mp._fields)
    missing_fields = expected_fields - actual
    if missing_fields:
        print(f"[verify_setup] mp missing fields: {missing_fields}")
        sys.exit(1)

    if mp.S.shape[1] != mp.Nc:
        print(f"[verify_setup] mp.S last dim {mp.S.shape[1]} != Nc {mp.Nc}")
        sys.exit(1)

    print(f"[verify_setup] OK — run_dir={run_dir}")
    print(f"[verify_setup] mp: Nc={mp.Nc} Ng_fit1={mp.Ng_fit1} Ng_fit2={mp.Ng_fit2}")
    print(f"[verify_setup] device={mp.device}")


if __name__ == "__main__":
    main()
