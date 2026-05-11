"""Section 4 — visualisation helpers.

Notebook cells 12 (UMAP / scatter / shift / widget helpers), 13 (fit-quality
overrides — 6-genes-per-row layout), 14 (log-scale S/U variant).

Widgets (e.g. `make_SP_shift_widget`) are kept for notebook use; skill
entrypoints call the underlying static plotting functions instead.
"""

import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import ipywidgets as widgets


# ── tiny helpers ─────────────────────────────────────────────────────────────

def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ── UMAP helpers ─────────────────────────────────────────────────────────────

def umap_color(adata, keys, title_prefix="", ncols=None, **kwargs):
    """Plot one UMAP panel per key in *keys*; kwargs forwarded to sc.pl.umap."""
    n = len(keys)
    ncols = ncols or n
    for key in keys:
        sc.pl.umap(adata, color=key, title=f"{title_prefix}{key}",
                   cmap=plt.cm.Spectral, **kwargs)


def umap_highlight_cluster(adata, cluster_key, cluster_value, **kwargs):
    """Highlight a single cluster value on UMAP, grey-out the rest."""
    tmp = adata.obs[cluster_key].copy().astype(object)
    tmp[tmp != cluster_value] = np.nan
    adata = adata.copy()
    adata.obs["_highlight"] = tmp
    sc.pl.umap(adata, color="_highlight",
               title=f"{cluster_key} = {cluster_value}",
               cmap=plt.cm.Spectral, **kwargs)


# ── scatter: observed vs. fitted counts along pseudotime ─────────────────────

def plot_phi_vs_counts(phi, elog_matrix, gene_names, gene_list,
                       obs_matrix=None, figsize_per_gene=(4, 3), ncols=6):
    """Scatter (ϕ, expected_counts) per gene; optionally overlay observed counts."""
    phi_np = phi.squeeze().detach().cpu().numpy() if hasattr(phi, 'detach') else np.asarray(phi).squeeze()
    n = len(gene_list)
    nr = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nr, min(n, ncols),
                             figsize=(figsize_per_gene[0] * min(n, ncols),
                                      figsize_per_gene[1] * nr),
                             squeeze=False)
    axes_flat = axes.flatten()

    for idx, g in enumerate(gene_list):
        gi = np.where(gene_names == g)[0][0]
        ax = axes_flat[idx]
        if obs_matrix is not None:
            y_obs = np.asarray(obs_matrix[gi]).squeeze()
            if hasattr(y_obs, 'numpy'):
                y_obs = y_obs.numpy()
            ax.scatter(phi_np, y_obs, s=6, c="grey", alpha=0.3, label="obs")
        y_fit = np.exp(np.asarray(elog_matrix[gi]).squeeze())
        if hasattr(y_fit, 'numpy'):
            y_fit = y_fit.numpy()
        ax.scatter(phi_np, y_fit, s=6, c="steelblue", label="fit")
        ax.set_title(g); ax.set_xlabel("ϕ"); ax.set_ylabel("counts")
        if obs_matrix is not None:
            ax.legend(fontsize=7, markerscale=2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


# ── fit-quality 6-genes/row layout (cell 13 overrides cell 12 versions) ──────

def plot_P_fits(phi, mp, ElogP_fit, ElogP2_fit, gene_names, gene_list,
                figsize_per_gene=(4, 3), ncols=6):
    """Protein fit quality: each gene is one column, rows are raw/post/true."""
    phi_np = _to_numpy(phi).squeeze()
    P_obs = _to_numpy(mp.P)
    P_post = np.exp(_to_numpy(ElogP_fit))
    P_true = np.exp(_to_numpy(ElogP2_fit))

    row_names = ["Raw P", "Posterior output P", "Posterior true P"]
    figs = []
    for start in range(0, len(gene_list), ncols):
        genes_chunk = list(gene_list[start:start + ncols])
        cols = len(genes_chunk)
        fig, axes = plt.subplots(
            len(row_names), cols,
            figsize=(figsize_per_gene[0] * cols, figsize_per_gene[1] * len(row_names)),
            squeeze=False, sharex=True,
        )

        for c, g in enumerate(genes_chunk):
            gi = np.where(gene_names == g)[0][0]
            y_curves = [P_obs[gi, :].squeeze(), P_post[gi, :].squeeze(), P_true[gi, :].squeeze()]
            ymax = max(float(np.nanmax(y)) for y in y_curves)
            ytop = 1.05 * ymax if ymax > 0 else 1.0

            for r, y in enumerate(y_curves):
                ax = axes[r, c]
                ax.scatter(phi_np, y, s=6, c="royalblue", alpha=0.30 if r == 0 else 0.85)
                ax.set_ylim(0, ytop)
                if r == 0:
                    ax.set_title(g)
                if c == 0:
                    ax.set_ylabel(f"{row_names[r]}\ncounts")
                if r == len(row_names) - 1:
                    ax.set_xlabel("ϕ")
        plt.tight_layout()
        figs.append(fig)
    return figs


def plot_SU_fits(phi, mp, ElogS_fit, ElogU_fit, ElogS2_fit, ElogU2_fit,
                 gene_names, gene_list, include_observable=True,
                 figsize_per_gene=(4, 3), ncols=6):
    """S/U fit quality: each gene is one column, S/U + raw/post/true are rows."""
    phi_np = _to_numpy(phi).squeeze()

    S_obs = _to_numpy(mp.S)
    U_obs = _to_numpy(mp.U)
    S_post = np.exp(_to_numpy(ElogS_fit))
    U_post = np.exp(_to_numpy(ElogU_fit))
    S_true = np.exp(_to_numpy(ElogS2_fit))
    U_true = np.exp(_to_numpy(ElogU2_fit))

    if include_observable:
        row_specs = [
            ("Raw S", "S", "obs", "blue", 0.30),
            ("Posterior output S", "S", "post", "blue", 0.85),
            ("Posterior true S", "S", "true", "blue", 0.85),
            ("Raw U", "U", "obs", "red", 0.30),
            ("Posterior output U", "U", "post", "red", 0.85),
            ("Posterior true U", "U", "true", "red", 0.85),
        ]
    else:
        row_specs = [
            ("Posterior output S", "S", "post", "blue", 0.85),
            ("Posterior true S", "S", "true", "blue", 0.85),
            ("Posterior output U", "U", "post", "red", 0.85),
            ("Posterior true U", "U", "true", "red", 0.85),
        ]

    figs = []
    for start in range(0, len(gene_list), ncols):
        genes_chunk = list(gene_list[start:start + ncols])
        cols = len(genes_chunk)
        fig, axes = plt.subplots(
            len(row_specs), cols,
            figsize=(figsize_per_gene[0] * cols, figsize_per_gene[1] * len(row_specs)),
            squeeze=False, sharex=True,
        )

        for c, g in enumerate(genes_chunk):
            gi = np.where(gene_names == g)[0][0]
            s_tracks = [S_post[gi, :].squeeze(), S_true[gi, :].squeeze()]
            u_tracks = [U_post[gi, :].squeeze(), U_true[gi, :].squeeze()]
            if include_observable:
                s_tracks = [S_obs[gi, :].squeeze()] + s_tracks
                u_tracks = [U_obs[gi, :].squeeze()] + u_tracks

            ymax_s = max(float(np.nanmax(y)) for y in s_tracks)
            ymax_u = max(float(np.nanmax(y)) for y in u_tracks)
            ytop_s = 1.05 * ymax_s if ymax_s > 0 else 1.0
            ytop_u = 1.05 * ymax_u if ymax_u > 0 else 1.0

            for r, (row_name, kind, source, color, alpha) in enumerate(row_specs):
                ax = axes[r, c]
                if kind == "S":
                    y = {"obs": S_obs, "post": S_post, "true": S_true}[source][gi, :].squeeze()
                    ax.set_ylim(0, ytop_s)
                else:
                    y = {"obs": U_obs, "post": U_post, "true": U_true}[source][gi, :].squeeze()
                    ax.set_ylim(0, ytop_u)
                ax.scatter(phi_np, y, s=6, c=color, alpha=alpha)
                if r == 0:
                    ax.set_title(g)
                if c == 0:
                    ax.set_ylabel(f"{row_name}\ncounts")
                if r == len(row_specs) - 1:
                    ax.set_xlabel("ϕ")
        plt.tight_layout()
        figs.append(fig)
    return figs


def plot_SU_fits_log(phi, mp, ElogS_fit, ElogU_fit, ElogS2_fit, ElogU2_fit,
                     gene_names, gene_list, include_observable=True,
                     figsize_per_gene=(4, 3), ncols=6):
    """Log-scale variant of plot_SU_fits (cell 14)."""
    phi_np = _to_numpy(phi).squeeze()

    S_obs = np.log(_to_numpy(mp.S))
    U_obs = np.log(_to_numpy(mp.U))
    S_post = _to_numpy(ElogS_fit)
    U_post = _to_numpy(ElogU_fit)
    S_true = _to_numpy(ElogS2_fit)
    U_true = _to_numpy(ElogU2_fit)

    if include_observable:
        row_specs = [
            ("Raw S", "S", "obs", "blue", 0.30),
            ("Posterior output S", "S", "post", "blue", 0.85),
            ("Posterior true S", "S", "true", "blue", 0.85),
            ("Raw U", "U", "obs", "red", 0.30),
            ("Posterior output U", "U", "post", "red", 0.85),
            ("Posterior true U", "U", "true", "red", 0.85),
        ]
    else:
        row_specs = [
            ("Posterior output S", "S", "post", "blue", 0.85),
            ("Posterior true S", "S", "true", "blue", 0.85),
            ("Posterior output U", "U", "post", "red", 0.85),
            ("Posterior true U", "U", "true", "red", 0.85),
        ]

    figs = []
    for start in range(0, len(gene_list), ncols):
        genes_chunk = list(gene_list[start:start + ncols])
        cols = len(genes_chunk)
        fig, axes = plt.subplots(
            len(row_specs), cols,
            figsize=(figsize_per_gene[0] * cols, figsize_per_gene[1] * len(row_specs)),
            squeeze=False, sharex=True,
        )

        for c, g in enumerate(genes_chunk):
            gi = np.where(gene_names == g)[0][0]
            s_tracks = [S_post[gi, :].squeeze(), S_true[gi, :].squeeze()]
            u_tracks = [U_post[gi, :].squeeze(), U_true[gi, :].squeeze()]
            if include_observable:
                s_tracks = [S_obs[gi, :].squeeze()] + s_tracks
                u_tracks = [U_obs[gi, :].squeeze()] + u_tracks

            ymax_s = max(float(np.nanmax(y)) for y in s_tracks)
            ymax_u = max(float(np.nanmax(y)) for y in u_tracks)
            ytop_s = 1.05 * ymax_s if ymax_s > 0 else 1.0
            ytop_u = 1.05 * ymax_u if ymax_u > 0 else 1.0

            for r, (row_name, kind, source, color, alpha) in enumerate(row_specs):
                ax = axes[r, c]
                if kind == "S":
                    y = {"obs": S_obs, "post": S_post, "true": S_true}[source][gi, :].squeeze()
                    ax.set_ylim(0, ytop_s)
                else:
                    y = {"obs": U_obs, "post": U_post, "true": U_true}[source][gi, :].squeeze()
                    ax.set_ylim(0, ytop_u)
                ax.scatter(phi_np, y, s=6, c=color)
                if r == 0:
                    ax.set_title(g)
                if c == 0:
                    ax.set_ylabel(f"{row_name}\ncounts")
                if r == len(row_specs) - 1:
                    ax.set_xlabel("ϕ")
        plt.tight_layout()
        figs.append(fig)
    return figs


# ── velocity / drift diagnostics ─────────────────────────────────────────────

def plot_velocity_drift_scatter(drift, vel, xlabel, ylabel="||v||₂",
                                title=None, c=None, cbar_label=None, figsize=(6, 6)):
    """Scatter of two per-cell norms with optional colour coding; returns Pearson r."""
    rho = np.corrcoef(drift, vel)[0, 1]
    fig, ax = plt.subplots(figsize=figsize)
    sc_kw = dict(s=6, alpha=0.6)
    if c is not None:
        sc_kw.update(c=c, cmap="plasma_r", s=28, alpha=0.8)
    scat = ax.scatter(drift, vel, **sc_kw)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title or f"Pearson r = {rho:.2f}")
    if c is not None and cbar_label:
        plt.colorbar(scat, ax=ax, label=cbar_label)
    plt.tight_layout()
    return rho, fig


def plot_velocity_alignment(vel_gc, drift_gc, vel_cell, drift_cell):
    """Gene–cell scatter + per-cell cosine-similarity histogram."""
    cos_sim = (vel_gc * drift_gc).sum(0) / (vel_cell * drift_cell + 1e-12)
    abs_err = np.abs(vel_gc - drift_gc)

    print(f"max  |Δ| : {abs_err.max():.4f}  |  mean |Δ| : {abs_err.mean():.4f}")
    print(f"cosine sim — min: {cos_sim.min():.4f}  |  mean: {cos_sim.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(drift_gc.flatten(), vel_gc.flatten(), s=2, alpha=0.4)
    axes[0].set_xlabel("drift component  βU−γS")
    axes[0].set_ylabel("velocity component  ω·S·D")
    axes[0].set_title("Gene–cell components")
    axes[0].plot([0, 1], [0, 1], transform=axes[0].transAxes, c="red")
    axes[1].hist(cos_sim, bins=40)
    axes[1].set_xlabel("cosine(v, drift)")
    axes[1].set_title("Per-cell vector alignment")
    plt.tight_layout()
    return fig


def plot_umap_velocity(
    adata,
    vel_cell,
    drift_cell=None,
    vmax_pct=99,
    ax=None,
    drift_ax=None,
    show=True,
    vel_key="velocity_abs",
    drift_key="velocity_abs_drift",
    vel_title="Model velocity ||v||",
    drift_title="Drift ||βU − γS||",
):
    """UMAP coloured by velocity magnitude; if drift_cell given, also drift magnitude."""
    adata.obs[vel_key] = vel_cell
    vmax = np.percentile(vel_cell, vmax_pct)
    sc.pl.umap(
        adata,
        color=vel_key,
        vmin=0,
        vmax=vmax,
        cmap="viridis",
        size=30,
        frameon=False,
        title=vel_title,
        ax=ax,
        show=show,
    )
    if drift_cell is not None:
        adata.obs[drift_key] = drift_cell
        sc.pl.umap(
            adata,
            color=drift_key,
            vmin=0,
            vmax=vmax,
            cmap="viridis",
            size=30,
            frameon=False,
            title=drift_title,
            ax=drift_ax,
            show=show,
        )


# ── shift analysis ─────────────────────────────────────────────────────────────

def compute_SP_derivatives(S_norm, P_norm, order, dphi, window_size):
    """Return (dS, dP, phi_mid) windowed finite-difference derivatives."""
    def _deriv(arr):
        raw = (arr[order][window_size:] - arr[order][:-window_size]) / (dphi[:-(window_size - 1)].sum() if window_size > 1 else dphi)
        return np.concatenate([np.full(window_size, np.nan), raw])

    pseudotime_sorted = np.sort(order.astype(float))
    phi_mid = np.concatenate([np.full(window_size, np.nan),
                              ((pseudotime_sorted[:-1] + pseudotime_sorted[1:]) / 2)[window_size - 1:]])
    return _deriv(S_norm), _deriv(P_norm), phi_mid


def plot_gene_SP_shift(gene, S_dense, P_dense, gene_names_list,
                       pseudotime_sorted, order, dphi, window_size,
                       genes_ordered=None):
    """Two-panel S/P shift plot for one gene (raw values + windowed derivatives)."""
    gi = gene_names_list.index(gene)
    s_sorted = S_dense[gi, order]
    p_sorted = P_dense[gi, order]

    dS = (s_sorted[window_size:] - s_sorted[:-window_size]) / np.sum(dphi[:window_size])
    dP = (p_sorted[window_size:] - p_sorted[:-window_size]) / np.sum(dphi[:window_size])
    dS = np.concatenate([np.full(window_size, np.nan), dS])
    dP = np.concatenate([np.full(window_size, np.nan), dP])

    phi_mid = (pseudotime_sorted[:-1] + pseudotime_sorted[1:]) / 2
    phi_mid = np.concatenate([np.full(window_size, np.nan), phi_mid[window_size - 1:]])

    score = np.nanmean(np.abs(dS - dP))
    rank_str = f"rank #{genes_ordered.index(gene) + 1}  —  " if genes_ordered is not None else ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{gene}   ({rank_str}mean |ΔS/Δϕ − ΔP/Δϕ| = {score:.4f})", fontsize=12)

    ax_s = axes[0]; ax_p = axes[0].twinx()
    ax_s.scatter(pseudotime_sorted, s_sorted, s=3, alpha=0.35, color='steelblue', label='S (RNA)')
    ax_p.scatter(pseudotime_sorted, p_sorted, s=3, alpha=0.35, color='tomato', label='P (ADT)')
    ax_s.set_xlabel('Pseudotime (ϕ)'); ax_s.set_ylabel('S — model expectation', color='steelblue')
    ax_p.set_ylabel('P — model expectation', color='tomato')
    ax_s.tick_params(axis='y', labelcolor='steelblue'); ax_p.tick_params(axis='y', labelcolor='tomato')
    ax_s.set_title('S and P along pseudotime (dual scale)')
    lines_s, labs_s = ax_s.get_legend_handles_labels()
    lines_p, labs_p = ax_p.get_legend_handles_labels()
    ax_s.legend(lines_s + lines_p, labs_s + labs_p, fontsize=9, markerscale=3)

    ax_ds = axes[1]; ax_dp = axes[1].twinx()
    ax_ds.plot(phi_mid, dS, color='steelblue', lw=1.5, label='ΔS/Δϕ')
    ax_dp.plot(phi_mid, dP, color='tomato', lw=1.5, label='ΔP/Δϕ')
    ax_ds.axhline(0, color='black', lw=0.6, ls='--'); ax_dp.axhline(0, color='tomato', lw=0.4, ls=':')
    ax_ds.set_xlabel('Pseudotime (ϕ)'); ax_ds.set_ylabel('ΔS/Δϕ (windowed)', color='steelblue')
    ax_dp.set_ylabel('ΔP/Δϕ (windowed)', color='tomato')
    ax_ds.tick_params(axis='y', labelcolor='steelblue'); ax_dp.tick_params(axis='y', labelcolor='tomato')
    ax_ds.set_title('Rate of variation (dual scale)')
    lines_ds, labs_ds = ax_ds.get_legend_handles_labels()
    lines_dp, labs_dp = ax_dp.get_legend_handles_labels()
    ax_ds.legend(lines_ds + lines_dp, labs_ds + labs_dp, fontsize=9)

    fig.tight_layout()
    return fig


def plot_shift_barplot(results_df, window_size):
    """Horizontal bar-chart of gene divergence scores."""
    fig = plt.figure(figsize=(10, max(4, len(results_df) * 0.35)))
    plt.barh(results_df['gene'][::-1], results_df['mean_|dS/dϕ - dP/dϕ|'][::-1], color='steelblue')
    plt.xlabel('Mean |dS/dϕ − dP/dϕ|')
    plt.title(f'Divergence: RNA vs ADT derivative  (window = {window_size})')
    plt.tight_layout()
    return fig


def make_SP_shift_widget(S_dense, P_dense, gene_names_list,
                         pseudotime_sorted, order, dphi, window_size, genes_ordered):
    """Interactive widget: dropdown → plot_gene_SP_shift for the selected gene."""
    dropdown = widgets.Dropdown(
        options=genes_ordered,
        value=genes_ordered[0],
        description='Gene:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='280px'),
    )
    out = widgets.Output()

    def _on_change(change):
        out.clear_output(wait=True)
        with out:
            plot_gene_SP_shift(change['new'], S_dense, P_dense,
                               gene_names_list, pseudotime_sorted, order, dphi,
                               window_size, genes_ordered)

    dropdown.observe(_on_change, names='value')
    with out:
        plot_gene_SP_shift(genes_ordered[0], S_dense, P_dense,
                           gene_names_list, pseudotime_sorted, order, dphi,
                           window_size, genes_ordered)
    return widgets.VBox([dropdown, out])
