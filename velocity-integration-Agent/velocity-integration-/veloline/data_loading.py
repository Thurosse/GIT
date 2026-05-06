"""Section 5 — data loading: read .h5ad, optional barcode (HTO) demultiplexing, ADT→RNA map.

Notebook cells 16, 18, 20.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def load_adata(data_path):
    """Read the AnnData .h5ad at *data_path* (cell 16)."""
    return sc.read(data_path)


def demultiplex_barcode(adata, barcode_names, barcode_selected):
    """Optional HTO/barcode demultiplexing (cell 18).

    If `barcode_selected` is None, returns adata unchanged.
    Otherwise, runs winner-takes-all on `adata.obsm["X_adt"]` columns matching
    `barcode_names`, filters cells assigned to `barcode_names[barcode_selected]`,
    and returns the filtered copy.
    """
    if barcode_selected is None:
        return adata

    barcode_list = list(barcode_names)
    chosen = barcode_list[barcode_selected]

    adt_names = list(adata.uns["adt_var_names"])
    missing = [b for b in barcode_list if b not in adt_names]
    if missing:
        raise ValueError(f"Some BARCODE_NAMES not in adt_var_names: {missing[:10]}")

    cols = [adt_names.index(b) for b in barcode_list]
    Xadt = adata.obsm["X_adt"][:, cols]

    if sparse.issparse(Xadt):
        X = Xadt.toarray()
    else:
        X = np.asarray(Xadt)

    winner_idx = X.argmax(axis=1)
    winner_name = np.array(barcode_list, dtype=object)[winner_idx]

    winner_score = X[np.arange(X.shape[0]), winner_idx]
    X_sorted = np.sort(X, axis=1)
    second_best = X_sorted[:, -2] if X.shape[1] >= 2 else np.zeros(X.shape[0])
    margin = winner_score - second_best

    keep = (winner_score > 0) & (margin >= 1)

    adata.obs["barcode_winner"] = winner_name
    adata.obs["barcode_winner_score"] = winner_score
    adata.obs["barcode_margin"] = margin
    adata.obs["barcode_assigned"] = keep

    selected_cells = (adata.obs["barcode_winner"].values == chosen) & (adata.obs["barcode_assigned"].values)
    print("chosen:", chosen)
    print("cells assigned to chosen:", int(selected_cells.sum()), "/", adata.n_obs)
    return adata[selected_cells].copy()


# ── ADT Protein → RNA Gene Symbol mapping (cell 20) ───────────────────────────

adt_rna_map = {
    'Hu.CD80': None,
    'Hu.CD86': 'CD86',
    'Hu.CD274': None,
    'Hu.CD273': None,
    'Hu.CD275': None,
    'Hu.CD155': None,
    'Hu.CD47': None,
    'Hu.CD70': 'CD70',
    'Hu.CD30': None,
    'Hu.CD48': 'CD48',
    'Hu.CD40': None,
    'Hu.CD154': None,
    'Hu.CD52': 'CD52',
    'Hu.CD3_UCHT1': None,
    'Hu.CD8': None,
    'Hu.CD56': None,
    'Hu.CD19': None,
    'Hu.CD33': 'CD33',
    'Hu.CD11c': 'ITGAM',
    'Hu.CD34': None,
    'Hu.CD138_MI15': None,
    'Hu.CD269': None,
    'Hu.HLA.ABC': None,
    'Hu.CD90': None,
    'Hu.CD117': None,
    'Hu.CD10': 'MME',
    'Hu.CD45RA': None,
    'Hu.CD123': None,
    'Hu.CD7': None,
    'Hu.CD105_43A3': None,
    'HuMs.CD49f': 'ITGA6',
    'Hu.CD194': None,
    'Hu.CD4_RPA.T4': None,
    'HuMs.CD44': 'CD44',
    'Hu.CD14_M5E2': 'CD14',
    'Hu.CD16': 'FCGR3A',
    'Hu.CD25': None,
    'Hu.CD45RO': None,
    'Hu.CD279': None,
    'Hu.TIGIT': None,
    'Hu.CD20_2H7': 'MS4A1',
    'Hu.CD335': None,
    'Hu.CD294': None,
    'Hu.CD326': None,
    'Hu.CD31': 'PECAM1',
    'Hu.Podoplanin': None,
    'Hu.EGFR': None,
    'Hu.IgM': None,
    'Hu.CD5': None,
    'Hu.CD183': None,
    'Hu.CD195': None,
    'Hu.CD32': 'FCGR2A',
    'Hu.CD196': None,
    'Hu.CD185': None,
    'Hu.CD103': None,
    'Hu.CD69': 'CD69',
    'Hu.CD62L': 'SELL',
    'Hu.CD197': 'CCR7',
    'Hu.CD161': None,
    'Hu.CD152': None,
    'Hu.CD223': None,
    'Hu.KLRG1': None,
    'Hu.CD27': None,
    'Hu.CD107a': None,
    'Hu.CD95': None,
    'Hu.CD134': None,
    'Hu.HLA.DR': 'HLA-DRA',
    'Hu.CD1c': 'CD1C',
    'Hu.CD11b': 'ITGAM',
    'Hu.CD64': 'FCGR1A',
    'Hu.CD141': None,
    'Hu.CD1d': 'CD1D',
    'Hu.CD314': 'KLRK1',
    'Hu.CD66b': None,
    'Hu.CD35': None,
    'Hu.CD57': None,
    'Hu.CD366': None,
    'Hu.CD272': None,
    'HuMsRt.CD278': None,
    'Hu.CD58': 'CD58',
    'Hu.CD96': None,
    'Hu.CD39': None,
    'Hu.CX3CR1': None,
    'Hu.CD24': 'CD24',
    'Hu.CD21': None,
    'Hu.CD11a': None,
    'Hu.CD79b': None,
    'Hu.CD244': None,
    'Hu.CD235ab': None,
    'Hu.Siglec.8': None,
    'Hu.CD206': 'MRC1',
    'Hu.CD169': None,
    'Hu.CD268': 'TNFRSF13B',
    'Hu.CD54': 'ICAM1',
    'Hu.CD62P': None,
    'Hu.CD119': 'IFNGR1',
    'Hu.TCR.AB': None,
    'Hu.CD68': 'CD68',
    'Hu.CD192': None,
    'Hu.CD106': None,
    'Hu.CD122': None,
    'Hu.CD267': 'TNFRSF13B',
    'Hu.CD135': 'FLT3',
    'Hu.FceRIa': None,
    'Hu.CD41': 'ITGA2B',
    'Hu.CD137': 'TNFSF9',
    'Hu.CD43': None,
    'Hu.CD163': 'CD163',
    'Hu.CD83': 'CD83',
    'Hu.CD357': None,
    'Hu.CD59': 'CD59',
    'Hu.CD124': None,
    'Hu.CD13': None,
    'Hu.CD184': None,
    'Hu.CD2': None,
    'Hu.CD226_11A8': None,
    'Hu.CD303': None,
    'Hu.CD61': 'ITGB3',
    'Hu.CD81': None,
    'Hu.IgD': None,
    'Hu.CD18': 'ITGB2',
    'Hu.CD28': None,
    'Hu.CD38_HIT2': None,
    'Hu.CD127': 'IL7R',
    'Hu.CD45_HI30': None,
    'Hu.CD22': None,
    'Hu.CD26': None,
    'Hu.CD193': None,
    'Hu.CD63': 'CD63',
    'Hu.CD304': 'NRP1',
    'Hu.CD36': 'CD36',
    'Hu.CD158': None,
    'Hu.CD93': 'CD93',
    'Hu.CD200': None,
    'Hu.CD49a': None,
    'Hu.CD49d': None,
    'Hu.CD73': 'NT5E',
    'Hu.CD9': 'CD9',
    'Hu.CD209': None,
    'Hu.CD337': None,
    'Hu.CD336': None,
    'Hu.CD186': None,
    'Hu.CD99': None,
    'Hu.CLEC12A': 'CLEC12A',
    'Hu.CD151': None,
    'Hu.CLEC1B': None,
    'Hu.CD94': None,
    'Hu.CD84': None,
    'Hu.CD23': None,
    'Hu.GPR56': 'ADGRG1',
    'Hu.CD82': 'CD82',
    'Hu.NKp80': None,
    'Hu.HLA.DR.DP.DQ': 'HLA-DRA',
    'Hu.CD181': None,
    'Hu.CD85k_ILT3': 'LILRB4',
    'Hu.CD85d_ILT4': 'LILRB2',
    'Hu.CD227_MUC1': None,
    'Hu.CD191_CCR1': None,
    'Hu.CD312_EMR2': 'ADGRE2',
    'Hu.CD49e': 'ITGA5',
    'Hu.CD159a_NKG2A': None,
    'Hu.CD159c_NKG2C': None,
    'Hu.CD15': None,
}

df_mapping = pd.DataFrame(list(adt_rna_map.items()), columns=['ADT_Protein', 'RNA_Marker'])
adt_list = df_mapping["RNA_Marker"]
adt_to_rna = df_mapping.copy()
