"""Sanity-check that inference outputs exist and have the expected keys."""

import os
import sys

import torch

from veloline import io_state


def main():
    run_dir = io_state.latest_run_dir()
    state = os.path.join(run_dir, "state")
    metrics = os.path.join(run_dir, "metrics")

    fit1 = os.path.join(state, "fit1_posteriors.pt")
    fit2 = os.path.join(state, "fit2_posteriors.pt")

    missing = [p for p in (fit1, fit2) if not os.path.exists(p)]
    if missing:
        print(f"[verify_inference] missing posterior files: {missing}")
        sys.exit(1)

    p1 = torch.load(fit1, map_location="cpu", weights_only=False)
    p2 = torch.load(fit2, map_location="cpu", weights_only=False)

    fit1_required = {"ϕ_fit", "ν_fit1"}
    if not fit1_required.issubset(p1.keys()):
        print(f"[verify_inference] fit1 missing keys: {fit1_required - set(p1.keys())}")
        sys.exit(1)

    fit2_required = {"ϕ_fit", "βg_fit", "γg_fit", "ElogP_fit", "ElogS_fit", "ElogU_fit",
                     "ElogP2_fit", "ElogS2_fit", "ElogU2_fit"}
    if not fit2_required.issubset(p2.keys()):
        print(f"[verify_inference] fit2 missing keys: {fit2_required - set(p2.keys())}")
        sys.exit(1)

    elbo_csv = os.path.join(metrics, "fit2_elbo.csv")
    if not (os.path.exists(elbo_csv) and os.path.getsize(elbo_csv) > 0):
        print("[verify_inference] fit2_elbo.csv missing or empty (SVI runs only)")

    print(f"[verify_inference] OK — run_dir={run_dir}")
    print(f"[verify_inference] ϕ range: [{p2['ϕ_fit'].min():.3f}, {p2['ϕ_fit'].max():.3f}]")
    print(f"[verify_inference] β median: {p2['βg_fit'].exp().median().item():.3f}, γ median: {p2['γg_fit'].median().item():.3f}")


if __name__ == "__main__":
    main()
