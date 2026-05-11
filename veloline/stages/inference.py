"""Stage 2 — run FIT 1 and/or FIT 2 (notebook §9–10).

Run:
    python -m veloline.stages.inference [--run latest|NAME] [--stages fit1,fit2] [--force]

Reads state/{adata_fit.h5ad, mp.pt, model_workflow.json, rng_state.pt} from the
chosen run dir, runs inference, and persists posteriors + ELBO traces + plots.
"""

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from veloline.metaparams import USE_GPU, MODEL_WORKFLOW, FIT_1_TO_FIT_2_CONSTANTS
from veloline import io_state
from veloline.fit1 import run_fit1
from veloline.fit2 import run_fit2


def _save_elbo(res, path):
    """Write iter,elbo CSV from a FitResults `res["res"].losses`."""
    if res.get("backend") != "svi":
        return
    losses = getattr(res.get("res"), "losses", []) or []
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "elbo"])
        for i, v in enumerate(losses):
            w.writerow([i, float(v)])


def _plot_elbo(res, path):
    if res.get("backend") != "svi":
        return
    losses = getattr(res.get("res"), "losses", []) or []
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses); ax.set_yscale("log")
    ax.set_xlabel("iter"); ax.set_ylabel("ELBO loss"); ax.set_title("SVI ELBO")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--stages", default="fit1,fit2",
                    help="Comma-separated subset of {fit1,fit2}; honors RUN_FIT_1/RUN_FIT_2 in metaparams.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run a stage even if its output file already exists.")
    args = ap.parse_args()

    run_dir = io_state.resolve_run_dir(args.run)
    state_dir = os.path.join(run_dir, "state")
    plots_root = os.path.join(run_dir, "plots")
    metrics_root = os.path.join(run_dir, "metrics")
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(metrics_root, exist_ok=True)
    print(f"[inference] run_dir = {run_dir}")

    rng_state = io_state.load_rng_state(run_dir)
    if rng_state["use_gpu"] != USE_GPU:
        raise RuntimeError(
            f"USE_GPU mismatch: setup wrote use_gpu={rng_state['use_gpu']} "
            f"but current metaparams.USE_GPU={USE_GPU}. Re-run setup or restore the original flag."
        )
    io_state.restore_rng(rng_state)

    mp = io_state.load_mp(run_dir)
    with open(os.path.join(state_dir, "model_workflow.json"), "r", encoding="utf-8") as f:
        workflow = json.load(f)
    fit1_cfg = workflow.get("fit_1", MODEL_WORKFLOW["fit_1"])
    fit2_cfg = workflow.get("fit_2", MODEL_WORKFLOW["fit_2"])
    fit2_cfg.setdefault("fit1_constants", list(FIT_1_TO_FIT_2_CONSTANTS))

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    fit1_post_path = os.path.join(state_dir, "fit1_posteriors.pt")
    fit1_pps_path = os.path.join(state_dir, "fit1_pps.pt")
    fit2_post_path = os.path.join(state_dir, "fit2_posteriors.pt")

    fit1_posteriors = None
    pps_fit1 = None

    # ── FIT 1 ─────────────────────────────────────────────────────────────────
    if "fit1" in requested and fit1_cfg.get("enabled", True):
        if os.path.exists(fit1_post_path) and not args.force:
            print(f"[inference] fit1: posteriors already exist at {fit1_post_path}; skipping (use --force to re-run)")
            fit1_posteriors = io_state.load_posteriors(fit1_post_path)
            if os.path.exists(fit1_pps_path):
                pps_fit1 = io_state.load_posteriors(fit1_pps_path)
        else:
            t0 = time.time()
            io_state.append_log(run_dir, "inference", "fit1: starting")
            mp_fit1, res_fit1, pps_fit1, fit1_posteriors = run_fit1(mp, fit1_cfg)
            io_state.save_posteriors(fit1_posteriors, fit1_post_path)
            io_state.save_posteriors(pps_fit1, fit1_pps_path)
            _save_elbo(res_fit1, os.path.join(metrics_root, "fit1_elbo.csv"))
            os.makedirs(os.path.join(plots_root, "fit1"), exist_ok=True)
            _plot_elbo(res_fit1, os.path.join(plots_root, "fit1", "elbo.png"))
            io_state.append_log(run_dir, "inference", f"fit1: done in {time.time()-t0:.1f}s")
            io_state.save_rng_state(run_dir, seed=rng_state["pyro_seed"], use_gpu=USE_GPU)

    elif os.path.exists(fit1_post_path):
        # Always load existing fit1 posteriors so fit2 can use them
        fit1_posteriors = io_state.load_posteriors(fit1_post_path)
        if os.path.exists(fit1_pps_path):
            pps_fit1 = io_state.load_posteriors(fit1_pps_path)

    # ── FIT 2 ─────────────────────────────────────────────────────────────────
    if "fit2" in requested and fit2_cfg.get("enabled", True):
        if os.path.exists(fit2_post_path) and not args.force:
            print(f"[inference] fit2: posteriors already exist at {fit2_post_path}; skipping (use --force to re-run)")
        elif pps_fit1 is None and fit1_cfg.get("enabled", True):
            raise RuntimeError(
                "fit2 requested but no fit1 posterior-predictive samples available. "
                "Run fit1 first or provide an existing fit1_pps.pt in state/."
            )
        else:
            t0 = time.time()
            io_state.append_log(run_dir, "inference", "fit2: starting")
            ElogS2_fit1 = fit1_posteriors.get("ElogS2_fit1") if fit1_posteriors else None
            ElogU2_fit1 = fit1_posteriors.get("ElogU2_fit1") if fit1_posteriors else None
            mp_fit2, res_fit2, pps_fit2, fit2_posteriors = run_fit2(
                mp, fit1_cfg, fit2_cfg, pps_fit1,
                ElogS2_fit1=ElogS2_fit1, ElogU2_fit1=ElogU2_fit1,
            )
            io_state.save_posteriors(fit2_posteriors, fit2_post_path)
            io_state.save_posteriors(pps_fit2, os.path.join(state_dir, "fit2_pps.pt"))
            _save_elbo(res_fit2, os.path.join(metrics_root, "fit2_elbo.csv"))
            os.makedirs(os.path.join(plots_root, "fit2"), exist_ok=True)
            _plot_elbo(res_fit2, os.path.join(plots_root, "fit2", "elbo.png"))
            io_state.append_log(run_dir, "inference", f"fit2: done in {time.time()-t0:.1f}s")
            io_state.save_rng_state(run_dir, seed=rng_state["pyro_seed"], use_gpu=USE_GPU)

    print(f"[inference] complete; outputs in {state_dir}")


if __name__ == "__main__":
    main()
