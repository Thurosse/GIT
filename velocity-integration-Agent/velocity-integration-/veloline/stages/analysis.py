"""Stage 3 — analysis: fit-quality, velocity, shift, co-expression (notebook §11–14).

Run:
    python -m veloline.stages.analysis [--run latest|NAME] [--sections SUBSET]

Reads state/{adata_norm.h5ad, mp.pt, fit1_posteriors.pt, fit2_posteriors.pt}
from the chosen run dir; renders plots into `plots/<section>/` and metrics
into `metrics/`. The `--sections` flag accepts a comma list of any of:
fit_quality, velocity, shift, coexpression.
"""

import argparse
import os
import time

from veloline.metaparams import USE_GPU
from veloline import io_state
from veloline.analysis import fit_quality, velocity, shift, coexpression


SECTION_FNS = {
    "fit_quality": fit_quality.run,
    "velocity": velocity.run,
    "shift": shift.run,
    "coexpression": coexpression.run,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--sections", default="fit_quality,velocity,shift,coexpression")
    args = ap.parse_args()

    run_dir = io_state.resolve_run_dir(args.run)
    state_dir = os.path.join(run_dir, "state")
    plots_root = os.path.join(run_dir, "plots")
    metrics_root = os.path.join(run_dir, "metrics")
    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(metrics_root, exist_ok=True)
    print(f"[analysis] run_dir = {run_dir}")

    rng_state = io_state.load_rng_state(run_dir)
    if rng_state["use_gpu"] != USE_GPU:
        raise RuntimeError(
            f"USE_GPU mismatch: setup wrote use_gpu={rng_state['use_gpu']} "
            f"but current metaparams.USE_GPU={USE_GPU}."
        )
    io_state.restore_rng(rng_state)

    mp = io_state.load_mp(run_dir)
    posteriors = io_state.load_posteriors(os.path.join(state_dir, "fit2_posteriors.pt"))
    pps_fit2_path = os.path.join(state_dir, "fit2_pps.pt")
    pps_fit2 = io_state.load_posteriors(pps_fit2_path) if os.path.exists(pps_fit2_path) else None

    adata_norm = io_state.load_adata(os.path.join(state_dir, "adata_norm.h5ad"))
    data_to_fit = io_state.load_adata(os.path.join(state_dir, "data_to_fit.h5ad"))

    requested = [s.strip() for s in args.sections.split(",") if s.strip()]
    saved_paths = []

    for sec in requested:
        if sec not in SECTION_FNS:
            print(f"[analysis] skipping unknown section: {sec}")
            continue
        plot_dir = os.path.join(plots_root, sec)
        t0 = time.time()
        io_state.append_log(run_dir, "analysis", f"{sec}: starting")
        if sec == "fit_quality":
            saved = SECTION_FNS[sec](data_to_fit, mp, posteriors, plot_dir)
        elif sec == "velocity":
            saved, _metrics = SECTION_FNS[sec](adata_norm, mp, posteriors, pps_fit2, plot_dir, metrics_root)
        elif sec == "shift":
            saved, _df = SECTION_FNS[sec](posteriors, mp, plot_dir, metrics_root)
        elif sec == "coexpression":
            saved = SECTION_FNS[sec](adata_norm, mp, posteriors, plot_dir, metrics_root)
        saved_paths.extend(saved or [])
        io_state.append_log(run_dir, "analysis", f"{sec}: done in {time.time()-t0:.1f}s ({len(saved or [])} files)")

    print(f"[analysis] wrote {len(saved_paths)} files under {plots_root}")


if __name__ == "__main__":
    main()
