"""§11 — fit-quality plots: P, S, U observable / posterior-output / posterior-true."""

import os
import numpy as np

from veloline.viz import plot_phi_vs_counts, plot_P_fits, plot_SU_fits


def run(data_to_fit, mp, posteriors, plot_dir, n_genes=6):
    """Generate the §11 panels and save PNGs to `plot_dir`.

    `posteriors` is the dict returned by `extract_fit2_posteriors`.
    Falls back to fit1 ElogS2_fit1 for a phi-vs-counts panel when fit2 is absent.
    """
    os.makedirs(plot_dir, exist_ok=True)
    gene_names_arr = np.array(data_to_fit.var.index)
    saved = []

    if posteriors.get("ElogS2_fit") is not None:
        fig = plot_phi_vs_counts(
            posteriors["ϕ_fit"],
            posteriors["ElogS2_fit"],
            gene_names_arr,
            gene_list=gene_names_arr[:12], ncols=6,
        )
        path = os.path.join(plot_dir, "phi_vs_counts.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        saved.append(path)

    if posteriors.get("ElogP_fit") is not None and posteriors.get("ElogP2_fit") is not None:
        figs = plot_P_fits(
            posteriors["ϕ_fit"], mp,
            posteriors["ElogP_fit"], posteriors["ElogP2_fit"],
            gene_names_arr, gene_list=gene_names_arr[:n_genes],
        )
        for i, fig in enumerate(figs):
            path = os.path.join(plot_dir, f"P_fits_{i:02d}.png")
            fig.savefig(path, dpi=140, bbox_inches="tight")
            saved.append(path)

    if all(posteriors.get(k) is not None for k in
           ("ElogS_fit", "ElogU_fit", "ElogS2_fit", "ElogU2_fit")):
        figs = plot_SU_fits(
            posteriors["ϕ_fit"], mp,
            posteriors["ElogS_fit"], posteriors["ElogU_fit"],
            posteriors["ElogS2_fit"], posteriors["ElogU2_fit"],
            gene_names_arr, gene_list=gene_names_arr[:n_genes],
            include_observable=True,
        )
        for i, fig in enumerate(figs):
            path = os.path.join(plot_dir, f"SU_fits_{i:02d}.png")
            fig.savefig(path, dpi=140, bbox_inches="tight")
            saved.append(path)

    return saved
