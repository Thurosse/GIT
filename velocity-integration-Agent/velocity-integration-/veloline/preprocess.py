"""Section 6 — pre-processing: cell filter, HVG, normalisation, neighbour graph,
size-factor normalisation, build adata_fit & data_to_fit.

Notebook cells 22–29.
"""

import numpy as np
import scanpy as sc

from veloline.data_loading import df_mapping
from veloline.metaparams import (
    EXCLUDED_CLUSTERS, cell_id,
    N_HIGHLY_VARIABLE,
    N_NEIGHBORS, N_PCS, N_DIFFMAP,
    MIN_CELLS_FRACTION, MIN_UNSPLICED_MEAN, MIN_SPLICED_MEAN,
    FORCE_INCLUDE_ADT_MAPPED_GENES, FORCED_ADT_PROTEINS,
)


def filter_clusters(adata):
    """6.1 — drop cells whose `cell_id` is in `EXCLUDED_CLUSTERS`."""
    n_before = adata.n_obs
    for cluster in EXCLUDED_CLUSTERS:
        adata = adata[adata.obs[cell_id] != cluster].copy()
    print(f"Cells: {n_before} → {adata.n_obs}  (excluded: {', '.join(EXCLUDED_CLUSTERS)})")
    return adata


def normalize_for_manifold(adata):
    """6.2 — HVG selection (in-place on `adata`) + log-normalised copy `adata_norm`
    with neighbours / diffmap / UMAP / DPT computed.
    Returns (adata_hvg, adata_norm, adata_fit_full).
    """
    adata_fit_full = adata.copy()
    if N_HIGHLY_VARIABLE > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=N_HIGHLY_VARIABLE, flavor='seurat_v3')
        adata = adata[:, adata.var['highly_variable']].copy()

    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm)
    sc.pp.log1p(adata_norm)

    sc.pp.neighbors(adata_norm, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)
    sc.tl.diffmap(adata_norm, n_comps=N_DIFFMAP)
    sc.tl.umap(adata_norm)
    sc.tl.dpt(adata_norm, n_dcs=2)
    return adata, adata_norm, adata_fit_full


def build_adata_fit(adata, adata_fit_full):
    """6.4 — gene filters + ADT-matched gene re-inclusion → returns adata_fit."""
    adata_fit_full.obs["batch"] = "pancreas_beta"
    adata_fit = adata.copy()

    sc.pp.filter_genes(adata_fit, min_cells=int(adata_fit.n_obs * MIN_CELLS_FRACTION))
    adata_fit = adata_fit[:, adata_fit.layers["unspliced"].toarray().mean(0) > MIN_UNSPLICED_MEAN].copy()
    adata_fit = adata_fit[:, adata_fit.layers["spliced"].toarray().mean(0) > MIN_SPLICED_MEAN].copy()

    genes_after_filters = set(adata_fit.var_names.tolist())

    _mapping = df_mapping[df_mapping["RNA_Marker"].notna()].copy()
    if len(FORCED_ADT_PROTEINS) > 0:
        _mapping = _mapping[_mapping["ADT_Protein"].isin(set(FORCED_ADT_PROTEINS))]

    matched_adt_rna_genes_all = sorted({
        str(rna) for _adt, rna in _mapping[["ADT_Protein", "RNA_Marker"]].itertuples(index=False, name=None)
    })

    matched_after_filters = sorted([g for g in genes_after_filters if g in set(matched_adt_rna_genes_all)])
    readded_genes = []
    if FORCE_INCLUDE_ADT_MAPPED_GENES:
        readded_genes = sorted(set(matched_adt_rna_genes_all) - set(matched_after_filters))

    final_matched_genes = sorted(set(readded_genes) | set(genes_after_filters))
    if len(final_matched_genes) == 0:
        raise ValueError("No ADT-matched RNA genes are available after applying filters/force-include policy.")

    mask_final = adata_fit_full.var_names.isin(final_matched_genes)
    adata_fit = adata_fit_full[:, mask_final].copy()

    print(f"Genes after filters (before ADT matching): {len(genes_after_filters)}")
    print(f"ADT-matched genes retained from filters: {len(matched_after_filters)}")
    print(f"ADT-mapped genes re-added by forcing: {len(readded_genes)}")
    print(f"Final ADT-matched genes for fitting: {adata_fit.n_vars}  |  Cells: {adata_fit.n_obs}")
    return adata_fit


def normalize_size_factors(adata_fit):
    """6.5 — add size-normalised layers `S_sz`, `U_sz` and `X_adt_sz` (in-place)."""
    for layer_key, obs_key, out_key in [("spliced", "n_scounts", "S_sz"),
                                        ("unspliced", "n_ucounts", "U_sz")]:
        counts = adata_fit.layers[layer_key].toarray().sum(1)
        adata_fit.obs[obs_key] = counts
        nf = np.mean(counts) / counts
        adata_fit.layers[out_key] = (nf * adata_fit.layers[layer_key].toarray().T).T

    if "X_adt" in adata_fit.obsm:
        pcounts = adata_fit.obsm["X_adt"].toarray().sum(1)
        adata_fit.obs["n_pcounts"] = pcounts
        pcc = np.where(pcounts == 0, 1.0, pcounts).astype(np.float32)
        nf = np.mean(pcc) / pcc
        adata_fit.obsm["X_adt_sz"] = (nf * adata_fit.obsm["X_adt"].toarray().T).T


def build_data_to_fit(adata_fit):
    """Cell 29 — final view used by §8 to build the mp container."""
    full_keep_genes = np.array(adata_fit.var.index)
    data_to_fit = adata_fit[:, [g in full_keep_genes for g in adata_fit.var.index]].copy()
    print(f"data_to_fit: {data_to_fit.n_vars} genes × {data_to_fit.n_obs} cells")
    return data_to_fit


def preprocess_pipeline(adata):
    """Drive all of §6: filter → HVG → adata_fit → normalise → data_to_fit.
    Returns (adata_fit, adata_norm, data_to_fit).
    """
    adata = filter_clusters(adata)
    adata, adata_norm, adata_fit_full = normalize_for_manifold(adata)
    adata_fit = build_adata_fit(adata, adata_fit_full)
    normalize_size_factors(adata_fit)
    data_to_fit = build_data_to_fit(adata_fit)
    return adata_fit, adata_norm, data_to_fit
