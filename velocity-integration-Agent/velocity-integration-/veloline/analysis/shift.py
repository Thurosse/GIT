"""§13 — shift analysis: windowed dS/dϕ vs dP/dϕ per gene."""

import os
import numpy as np
import pandas as pd

from veloline.metaparams import SHIFT_WINDOW_SIZE
from veloline.data_loading import adt_to_rna
from veloline.viz import plot_shift_barplot, plot_gene_SP_shift


def _minmax_norm(arr):
    mn = arr.min(axis=1, keepdims=True)
    mx = arr.max(axis=1, keepdims=True)
    return (arr - mn) / (mx - mn + 1e-12)


def prepare_shift_arrays(posteriors, mp, threshold=0):
    """Build (S_dense, P_dense, gene_names_list, order, pseudotime_sorted, dϕ).

    `threshold` is the minimum pseudotime (default 0; cells below are dropped).
    """
    ElogS2 = posteriors["ElogS2_fit"]
    ElogP2 = posteriors["ElogP2_fit"]

    S_dense = _minmax_norm(np.exp(
        ElogS2.detach().cpu().numpy() if hasattr(ElogS2, "detach") else np.array(ElogS2)
    ))
    P_dense = _minmax_norm(np.exp(
        ElogP2.detach().cpu().numpy() if hasattr(ElogP2, "detach") else np.array(ElogP2)
    ))

    pseudotime = posteriors["ϕ_fit"].squeeze().detach().cpu().numpy() if hasattr(posteriors["ϕ_fit"], "detach") else posteriors["ϕ_fit"].numpy()
    order_full = np.argsort(pseudotime)
    indices = np.where(pseudotime[order_full] > threshold)[0]
    order = order_full[indices]

    pseudotime_sorted = pseudotime[order]
    dϕ = np.diff(pseudotime_sorted)

    row_n = int(S_dense.shape[0])
    gene_names_list = None
    for attr in ["fit2_gene_names", "fit1_gene_names", "gene_names_all"]:
        vals = getattr(mp, attr, None)
        if vals is not None and len(vals) == row_n:
            gene_names_list = [str(v) for v in list(vals)]
            break
    if gene_names_list is None:
        raise ValueError(f"Could not align gene names to matrix rows ({row_n}).")

    return S_dense, P_dense, gene_names_list, order, pseudotime_sorted, dϕ


def score_shift(S_dense, P_dense, gene_names_list, order, dϕ, window_size=None):
    """Compute mean |dS/dϕ − dP/dϕ| per ADT-matched gene; returns sorted DataFrame."""
    w = int(window_size if window_size is not None else SHIFT_WINDOW_SIZE)
    n_cells = int(len(order))
    if w <= 0 or w >= n_cells:
        raise ValueError(f"SHIFT_WINDOW_SIZE must be in [1, {n_cells - 1}], got {w}")

    den = dϕ[:-(w - 1)].sum() if w > 1 else dϕ.sum()
    if not np.isfinite(den) or den == 0:
        raise ValueError(f"Invalid derivative denominator: {den}")

    gene_to_idx = {g: i for i, g in enumerate(gene_names_list)}
    if "RNA_Marker" in adt_to_rna.columns:
        rna_iter = adt_to_rna["RNA_Marker"].astype(str).tolist()
    else:
        rna_iter = [str(row[-1]) for row in adt_to_rna.itertuples(index=False, name=None)]

    results = []
    missing = 0
    for rna in rna_iter:
        gi = gene_to_idx.get(rna)
        if gi is None:
            missing += 1
            continue
        s_s = S_dense[gi, order]
        p_s = P_dense[gi, order]
        dS = (s_s[w:] - s_s[:-w]) / den
        dP = (p_s[w:] - p_s[:-w]) / den
        dS = np.concatenate([np.full(w, np.nan), dS])
        dP = np.concatenate([np.full(w, np.nan), dP])
        results.append((rna, np.nanmean(np.abs(dS - dP))))

    if not results:
        return pd.DataFrame(columns=["gene", "mean_|dS/dϕ - dP/dϕ|"])

    return (
        pd.DataFrame(results, columns=["gene", "mean_|dS/dϕ - dP/dϕ|"])
        .sort_values("mean_|dS/dϕ - dP/dϕ|", ascending=False)
        .reset_index(drop=True)
    )


def run(posteriors, mp, plot_dir, metrics_dir, n_top_genes=6):
    """Drive §13 end-to-end: scores, bar plot, and per-gene panels for the top genes."""
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    S_dense, P_dense, gene_names_list, order, pseudotime_sorted, dϕ = \
        prepare_shift_arrays(posteriors, mp, threshold=0)

    results_df = score_shift(S_dense, P_dense, gene_names_list, order, dϕ)
    results_df.to_csv(os.path.join(metrics_dir, "shift_scores.csv"), index=False)

    saved = []
    if len(results_df) > 0:
        fig = plot_shift_barplot(results_df, SHIFT_WINDOW_SIZE)
        p = os.path.join(plot_dir, "shift_barplot.png")
        fig.savefig(p, dpi=140, bbox_inches="tight")
        saved.append(p)

        genes_ordered = list(results_df["gene"])
        for gene in genes_ordered[:n_top_genes]:
            fig = plot_gene_SP_shift(
                gene, S_dense, P_dense, gene_names_list,
                pseudotime_sorted, order, dϕ, SHIFT_WINDOW_SIZE,
                genes_ordered,
            )
            p = os.path.join(plot_dir, f"shift_gene_{gene}.png")
            fig.savefig(p, dpi=140, bbox_inches="tight")
            saved.append(p)

    return saved, results_df
