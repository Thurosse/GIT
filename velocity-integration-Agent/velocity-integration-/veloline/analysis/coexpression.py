"""§14 — co-expression population analysis: lymphoid × myeloid RNA × ADT thresholded
co-expression on UMAP, plus KDE / raw-count histograms for the top population.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scanpy as sc


# ── Marker lists (cell 75) ────────────────────────────────────────────────────

LYMPHOID_MARKERS = [
    'MS4A1',     # CD20 (B cells)
    'IL7R',      # CD127 (T cells, especially memory/naive)
    'CCR7',      # T cells / B cells homing
    'KLRK1',     # CD314 / NKG2D
    'TNFRSF13B', # CD267/CD268 (B cells)
    'CD24',      # B cells maturation
    'CD69',      # Early activation
]

MYELOID_MARKERS = [
    'CD14',      # Monocytes classiques
    'CD33',      # Lignée myéloïde générale
    'ITGAM',     # CD11b/CD11c
    'FCGR1A',    # CD64
    'FCGR2A',    # CD32
    'CD163',     # Macrophages / Monocytes M2
    'MRC1',      # CD206
    'CLEC12A',   # DC
    'LILRB4',    # CD85k
    'CD68',      # Macrophages / Monocytes
    'CD36',      # Monocytes / Plaquettes
    'CD93',      # Early myeloid differentiation
]


def threshold_sweep(S_dense, P_dense, gene_names_list, order,
                    thresholds=None, top_n=20):
    """Cell 82 — sweep ADT/RNA thresholds and tally top RNA×ADT pairs per cell."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.2, 1.01, 0.1), 2)

    s_sorted = S_dense[:, order].copy()
    p_sorted = P_dense[:, order].copy()

    lymphoid_indices = [i for i, g in enumerate(gene_names_list) if g in LYMPHOID_MARKERS]
    myeloid_indices = [i for i, g in enumerate(gene_names_list) if g in MYELOID_MARKERS]
    if not lymphoid_indices or not myeloid_indices:
        raise ValueError("No marker genes found for one or both marker sets.")

    lymphoid_genes = [gene_names_list[i] for i in lymphoid_indices]
    myeloid_genes = [gene_names_list[i] for i in myeloid_indices]
    P_lymphoid = p_sorted[lymphoid_indices, :]
    S_myeloid = s_sorted[myeloid_indices, :]

    results = []
    for t in thresholds:
        adt_t = float(t)
        rna_t = float(t)
        P_f = np.where(P_lymphoid > adt_t, P_lymphoid, -np.inf)
        S_f = np.where(S_myeloid > rna_t, S_myeloid, -np.inf)
        l_valid = (P_f > -np.inf).any(axis=0)
        m_valid = (S_f > -np.inf).any(axis=0)
        both_valid = l_valid & m_valid

        l_top = np.where(l_valid, [lymphoid_genes[i] for i in np.argmax(P_f, axis=0)], "NONE")
        m_top = np.where(m_valid, [myeloid_genes[i] for i in np.argmax(S_f, axis=0)], "NONE")

        gene_pairs = [(l, m) for l, m, v in zip(l_top, m_top, both_valid) if v]
        pair_counts = Counter(gene_pairs)
        top_pairs = pair_counts.most_common(top_n)

        results.append({
            "threshold": float(t),
            "n_valid_cells": int(both_valid.sum()),
            "n_unique_pairs": len(pair_counts),
            "top_pairs": top_pairs,
        })
    return results, lymphoid_indices, myeloid_indices, lymphoid_genes, myeloid_genes


def overlay_top_pair_on_umap(adata_norm, posteriors, S_dense, P_dense, gene_names_list, order,
                             threshold=0.6, nb=1, plot_path=None):
    """Cell 84 — for the chosen `threshold`, find the most-common ADT×RNA pair (rank `nb`)
    and overlay the cells expressing both on the UMAP.

    Returns (most_common_pair, max_count, cells_with_pair_indices, fig).
    """
    s_sorted = S_dense[:, order].copy()
    p_sorted = P_dense[:, order].copy()

    lymphoid_indices = [i for i, g in enumerate(gene_names_list) if g in LYMPHOID_MARKERS]
    myeloid_indices = [i for i, g in enumerate(gene_names_list) if g in MYELOID_MARKERS]
    lymphoid_genes = [gene_names_list[i] for i in lymphoid_indices]
    myeloid_genes = [gene_names_list[i] for i in myeloid_indices]

    P_lymphoid = p_sorted[lymphoid_indices, :]
    S_myeloid = s_sorted[myeloid_indices, :]
    adt_t = rna_t = float(threshold)

    P_f = np.where(P_lymphoid > adt_t, P_lymphoid, -np.inf)
    S_f = np.where(S_myeloid > rna_t, S_myeloid, -np.inf)

    l_valid = (P_f > -np.inf).any(axis=0)
    m_valid = (S_f > -np.inf).any(axis=0)
    both_valid = l_valid & m_valid

    l_top = np.where(l_valid, [lymphoid_genes[i] for i in np.argmax(P_f, axis=0)], "NONE")
    m_top = np.where(m_valid, [myeloid_genes[i] for i in np.argmax(S_f, axis=0)], "NONE")

    gene_pairs = [(l, m) for l, m, v in zip(l_top, m_top, both_valid) if v]
    valid_cell_positions = np.where(both_valid)[0]
    pair_counts = Counter(gene_pairs)
    most_common = pair_counts.most_common(nb)
    if not most_common:
        return None, 0, [], None
    most_common_pair, max_count = most_common[nb - 1]

    cells_with_pair = [valid_cell_positions[i] for i, pair in enumerate(gene_pairs)
                       if pair == most_common_pair]
    l_idx = [order[elt] for elt in cells_with_pair]

    data_yann = adata_norm.copy()
    data_yann.obs["dpt_pseudotime"] = posteriors["ϕ_fit"].numpy()
    if isinstance(data_yann.obs.index[0], str):
        l_idx = data_yann.obs.index[l_idx]
    data_yann.obs["in_pair"] = "No"
    data_yann.obs.loc[l_idx, "in_pair"] = "Yes"

    fig = plt.figure(figsize=(6, 6))
    sc.pl.umap(
        data_yann, color="in_pair",
        palette=["lightgray", "red"],
        title=f"{most_common_pair[0]} × {most_common_pair[1]}  (n={max_count})",
        size=30, frameon=False, show=False,
    )
    if plot_path:
        plt.savefig(plot_path, dpi=140, bbox_inches="tight")
    return most_common_pair, max_count, cells_with_pair, fig


def kde_panels_for_top_pair(mp, S_dense, P_dense, gene_names_list, order, cells_with_pair,
                            most_common_pair, plot_dir):
    """Cell 86 — KDE / raw-count histograms for the top ADT×RNA pair."""
    cell_int_idx = [order[i] for i in cells_with_pair]
    cell_int_idx_arr = np.array(cell_int_idx, dtype=int)
    gene_idx = gene_names_list.index(most_common_pair[1])
    adt_idx = gene_names_list.index(most_common_pair[0])

    def _clean(arr, q=99):
        arr = np.asarray(arr).flatten()
        arr = arr[np.isfinite(arr)]
        return np.clip(arr, 0, np.percentile(arr, q))

    S_raw_all = mp.S[gene_idx, :].detach().cpu().numpy().flatten()
    S_raw_selected = mp.S[gene_idx, cell_int_idx_arr].detach().cpu().numpy().flatten()
    P_raw_all = mp.P[adt_idx, :].detach().cpu().numpy().flatten()
    P_raw_selected = mp.P[adt_idx, cell_int_idx_arr].detach().cpu().numpy().flatten()

    def _plot_kde(ax, all_vals, sel_vals, title, xlabel, n_zero_all=None, n_zero_sel=None):
        all_vals = np.asarray(all_vals).flatten()
        sel_vals = np.asarray(sel_vals).flatten()
        all_vals = all_vals[np.isfinite(all_vals)]
        sel_vals = sel_vals[np.isfinite(sel_vals)]
        if len(all_vals) == 0 or len(sel_vals) == 0:
            ax.set_title(f"{title}\n(no finite values)")
            return
        x = np.linspace(min(all_vals.min(), sel_vals.min()),
                        max(all_vals.max(), sel_vals.max()), 500)
        for vals, color, label in [
            (all_vals, "steelblue", f"all cells (n={len(all_vals)})"),
            (sel_vals, "tomato", f"selected (n={len(sel_vals)})"),
        ]:
            kde = gaussian_kde(vals, bw_method=0.35)
            ax.fill_between(x, kde(x), alpha=0.3, color=color)
            ax.plot(x, kde(x), color=color, lw=2, label=label)
        if n_zero_all is not None and n_zero_sel is not None:
            ax.annotate(f"zeros — all: {n_zero_all}  |  selected: {n_zero_sel}",
                        xy=(0.5, 0.97), xycoords="axes fraction",
                        ha="center", va="top", fontsize=8, color="gray", style="italic")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    saved = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_kde(axes[0], P_dense[adt_idx, :], P_dense[adt_idx, cell_int_idx],
              title=f"P_dense — {most_common_pair[0]}",
              xlabel="Normalised ADT expression",
              n_zero_all=(P_raw_all == 0).sum(), n_zero_sel=(P_raw_selected == 0).sum())
    _plot_kde(axes[1], S_dense[gene_idx, :], S_dense[gene_idx, cell_int_idx],
              title=f"S_dense — {most_common_pair[1]}",
              xlabel="Normalised RNA expression",
              n_zero_all=(S_raw_all == 0).sum(), n_zero_sel=(S_raw_selected == 0).sum())
    plt.suptitle(f"Model expectations — selected (n={len(cell_int_idx)}) vs all cells", fontsize=12)
    plt.tight_layout()
    p = os.path.join(plot_dir, "coexpr_kde_model_expectations.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_kde(axes[0], _clean(P_raw_all), _clean(P_raw_selected),
              title=f"mp.P (raw) — {most_common_pair[0]}",
              xlabel="Raw ADT counts",
              n_zero_all=(P_raw_all == 0).sum(), n_zero_sel=(P_raw_selected == 0).sum())
    _plot_kde(axes[1], _clean(S_raw_all), _clean(S_raw_selected),
              title=f"mp.S (raw) — {most_common_pair[1]}",
              xlabel="Raw RNA counts",
              n_zero_all=(S_raw_all == 0).sum(), n_zero_sel=(S_raw_selected == 0).sum())
    plt.suptitle(f"Raw counts — selected (n={len(cell_int_idx)}) vs all cells", fontsize=12)
    plt.tight_layout()
    p = os.path.join(plot_dir, "coexpr_kde_raw_counts.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mask_P = (P_raw_selected > 0)
    mask_S = (S_raw_selected > 0)
    if mask_P.any():
        axes[0].hist(P_raw_selected[mask_P], bins=np.arange(P_raw_selected[mask_P].max() + 2) - 0.5, alpha=0.5)
    axes[0].set_title(f"mp.P (raw) — {most_common_pair[0]}")
    axes[0].set_xlabel("Raw ADT counts"); axes[0].set_ylabel("Number of cells")
    if mask_S.any():
        axes[1].hist(S_raw_selected[mask_S], bins=np.arange(S_raw_selected[mask_S].max() + 2) - 0.5, alpha=0.5)
    axes[1].set_title(f"mp.S (raw) — {most_common_pair[1]}")
    axes[1].set_xlabel("Raw RNA counts"); axes[1].set_ylabel("Number of cells")
    plt.suptitle("Raw counts distribution (non-zero only)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(plot_dir, "coexpr_raw_count_hist.png")
    fig.savefig(p, dpi=140, bbox_inches="tight"); saved.append(p)
    return saved


def run(adata_norm, mp, posteriors, plot_dir, metrics_dir, threshold=0.6, nb=1):
    """Drive §14 end-to-end."""
    from veloline.analysis.shift import prepare_shift_arrays
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    S_dense, P_dense, gene_names_list, order, _, _ = prepare_shift_arrays(posteriors, mp, threshold=0)

    sweep, *_ = threshold_sweep(S_dense, P_dense, gene_names_list, order)
    rows = []
    for res in sweep:
        rows.append({
            "row_type": "summary",
            "threshold": res["threshold"],
            "rank": None,
            "lymphoid_adt": None,
            "myeloid_rna": None,
            "count": None,
            "n_valid_cells": res["n_valid_cells"],
            "n_unique_pairs": res["n_unique_pairs"],
        })
        for rank, (pair, count) in enumerate(res["top_pairs"], start=1):
            rows.append({
                "row_type": "pair",
                "threshold": res["threshold"],
                "rank": rank,
                "lymphoid_adt": pair[0],
                "myeloid_rna": pair[1],
                "count": count,
                "n_valid_cells": None,
                "n_unique_pairs": None,
            })
    if rows:
        cols = [
            "row_type",
            "threshold",
            "rank",
            "lymphoid_adt",
            "myeloid_rna",
            "count",
            "n_valid_cells",
            "n_unique_pairs",
        ]
        pd.DataFrame(rows, columns=cols).to_csv(
            os.path.join(metrics_dir, "coexpression_top20.csv"), index=False
        )

    saved = []
    most_common_pair, max_count, cells_with_pair, fig = overlay_top_pair_on_umap(
        adata_norm, posteriors, S_dense, P_dense, gene_names_list, order,
        threshold=threshold, nb=nb,
        plot_path=os.path.join(plot_dir, "coexpr_top_pair_umap.png"),
    )
    if fig is not None:
        saved.append(os.path.join(plot_dir, "coexpr_top_pair_umap.png"))

    if most_common_pair is not None:
        saved.extend(
            kde_panels_for_top_pair(
                mp, S_dense, P_dense, gene_names_list, order, cells_with_pair,
                most_common_pair, plot_dir,
            )
        )
    return saved
