"""§12 — velocity field: v = β·U − γ·S (RNA), spline-derivative drift (ADT),
Pearson alignment between RNA velocity and ADT drift.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from veloline.viz import (
    plot_umap_velocity, plot_velocity_drift_scatter, plot_velocity_alignment,
)


def compute_velocity_arrays(posteriors):
    """12.1–12.4 — compute (vel_gc, drift_gc, vel_cell, drift_cell) for RNA and ADT branches.

    Returns a dict with: vel_gc, drift_gc, vel_cell, drift_cell, vel_gc_p,
    drift_gc_p, vel_cell_p, drift_cell_p, S_hat, U_hat, P_hat.
    """
    βg = posteriors["βg_fit"]
    γg = posteriors["γg_fit"]
    Ag = posteriors["Ag_fit"]
    Bg = posteriors["Bg_fit"]
    ν = posteriors["ν_fit"]
    k_fit = posteriors["k_fit"]
    ζ_dϕ = posteriors["ζ_dϕ_fit"]
    ζ_d2ϕ = posteriors.get("ζ_d2ϕ_fit") if posteriors.get("ζ_d2ϕ_fit") is not None else ζ_dϕ

    β_hat = βg.detach().cpu().numpy() if hasattr(βg, "detach") else np.asarray(βg)
    γ_hat = γg.detach().cpu().numpy() if hasattr(γg, "detach") else np.asarray(γg)
    A_hat = Ag.detach().cpu().numpy() if hasattr(Ag, "detach") else np.asarray(Ag)
    B_hat = Bg.detach().cpu().numpy() if hasattr(Bg, "detach") else np.asarray(Bg)
    ω_hat = 1

    ζp = ζ_dϕ.cpu().numpy()
    ζp2 = ζ_d2ϕ.cpu().numpy()

    D = ν.cpu().numpy() @ ζp2.T
    D_p = k_fit.cpu().numpy() @ ζp.T

    S_hat = torch.exp(posteriors["ElogS_fit"]).cpu().numpy()
    U_hat = torch.exp(posteriors["ElogU_fit"]).cpu().numpy()
    P_hat = torch.exp(posteriors["ElogP_fit"]).cpu().numpy()

    vel_gc = ω_hat * D * S_hat
    drift_gc = β_hat[:, None] * U_hat - γ_hat[:, None] * S_hat
    vel_gc_p = ω_hat * D_p * P_hat
    drift_gc_p = B_hat[:, None] * S_hat - A_hat[:, None] * P_hat

    vel_cell = np.linalg.norm(vel_gc, axis=0)
    drift_cell = np.linalg.norm(drift_gc, axis=0)
    vel_cell_p = np.linalg.norm(vel_gc_p, axis=0)
    drift_cell_p = np.linalg.norm(drift_gc_p, axis=0)

    return dict(
        vel_gc=vel_gc, drift_gc=drift_gc,
        vel_gc_p=vel_gc_p, drift_gc_p=drift_gc_p,
        vel_cell=vel_cell, drift_cell=drift_cell,
        vel_cell_p=vel_cell_p, drift_cell_p=drift_cell_p,
        S_hat=S_hat, U_hat=U_hat, P_hat=P_hat,
    )


def run(adata_norm, mp, posteriors, pps_fit2, plot_dir, metrics_dir):
    """Drive §12.1–12.9 and persist plots + alignment metrics."""
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    arrays = compute_velocity_arrays(posteriors)
    saved = []

    # 12.5 UMAP overlays (single 2x2 panel)
    data_yann = adata_norm.copy()
    data_yann.obs["dpt_pseudotime"] = posteriors["ϕ_fit"].numpy()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plot_umap_velocity(
        data_yann,
        arrays["vel_cell"],
        arrays["drift_cell"],
        ax=axes[0, 0],
        drift_ax=axes[0, 1],
        show=False,
        vel_key="velocity_abs_rna",
        drift_key="velocity_abs_drift_rna",
        vel_title="RNA velocity ||v||",
        drift_title="RNA drift ||beta U - gamma S||",
    )
    plot_umap_velocity(
        data_yann,
        arrays["vel_cell_p"],
        arrays["drift_cell_p"],
        ax=axes[1, 0],
        drift_ax=axes[1, 1],
        show=False,
        vel_key="velocity_abs_adt",
        drift_key="velocity_abs_drift_adt",
        vel_title="ADT velocity ||v||",
        drift_title="ADT drift ||B S - A P||",
    )
    fig.tight_layout()
    p = os.path.join(plot_dir, "velocity_umap_intensity_2x2.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)
    plt.close(fig)

    # 12.6 RNA correlation
    pearson_rna = float(np.corrcoef(arrays["drift_cell"], arrays["vel_cell"])[0, 1])
    spearman_rna = float(spearmanr(arrays["drift_cell"], arrays["vel_cell"]).correlation)
    rho, fig = plot_velocity_drift_scatter(
        arrays["drift_cell"], arrays["vel_cell"],
        xlabel=r"$\|\beta U - \gamma S\|_2$",
        title=f"RNA — Pearson r = {pearson_rna:.2f}",
    )
    p = os.path.join(plot_dir, "rna_velocity_drift_scatter.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)

    # 12.7 per-cell uncertainty colouring (uses pps_fit2 samples)
    if pps_fit2 is not None and "logβg" in pps_fit2 and "logγg" in pps_fit2 and "ElogU" in pps_fit2:
        logβ_s = pps_fit2["logβg"].squeeze().cpu()
        logγ_s = pps_fit2["logγg"].squeeze().cpu()
        ElogU_s = pps_fit2["ElogU"].squeeze().cpu()

        S_hat_t = torch.exp(posteriors["ElogS_fit"]).cpu()
        U_hat_s = torch.exp(ElogU_s)
        β_s = torch.exp(logβ_s)[:, :, None]
        γ_s = torch.exp(logγ_s)[:, :, None]
        vel_mag_s = β_s * U_hat_s - γ_s * S_hat_t[None, :, :]
        vel_mag_mean = vel_mag_s.mean(0).T.cpu().numpy()
        vel_mag_sd = vel_mag_s.std(0).T.cpu().numpy()
        vel_unc = np.linalg.norm(vel_mag_sd, axis=1)
        vel_norm_exp = np.linalg.norm(vel_mag_mean, axis=1)

        rho2, fig2 = plot_velocity_drift_scatter(
            vel_norm_exp, arrays["vel_cell"],
            xlabel=r"$\|\hat\beta\hat U - \hat\gamma\hat S\|_2$",
            c=vel_unc, cbar_label=r"per-cell velocity SD",
        )
        p = os.path.join(plot_dir, "rna_velocity_drift_uncertainty.png")
        fig2.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)

    # 12.8 alignment
    fig3 = plot_velocity_alignment(
        arrays["vel_gc"], arrays["drift_gc"], arrays["vel_cell"], arrays["drift_cell"],
    )
    p = os.path.join(plot_dir, "rna_velocity_alignment.png")
    fig3.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)

    # 12.9 ADT correlation
    pearson_adt = float(np.corrcoef(arrays["drift_cell_p"], arrays["vel_cell_p"])[0, 1])
    spearman_adt = float(spearmanr(arrays["drift_cell_p"], arrays["vel_cell_p"]).correlation)
    rho4, fig4 = plot_velocity_drift_scatter(
        arrays["drift_cell_p"], arrays["vel_cell_p"],
        xlabel=r"$\|\hat B\hat S - \hat A\hat P\|_2$",
        title=f"ADT — Pearson r = {pearson_adt:.2f}",
    )
    p = os.path.join(plot_dir, "adt_velocity_drift_scatter.png")
    fig4.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)

    metrics = {
        "rna_pearson": pearson_rna,
        "rna_spearman": spearman_rna,
        "adt_pearson": pearson_adt,
        "adt_spearman": spearman_adt,
    }
    with open(os.path.join(metrics_dir, "velocity_alignment.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return saved, metrics
