#!/usr/bin/env python
import argparse
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.io import mmread

DEFAULT_SPLICED_KEYS = ["spliced", "Spliced", "S", "spliced_counts", "spliced_count", "counts_spliced"]
DEFAULT_UNSPLICED_KEYS = ["unspliced", "Unspliced", "U", "unspliced_counts", "unspliced_count", "counts_unspliced"]
DEFAULT_ADT_OBSM_KEYS = [
    "X_adt",
    "protein_expression",
    "X_protein",
    "ADT",
    "ADT_counts",
    "X_adt_counts",
    "citeseq",
    "X_citeseq",
]
DEFAULT_ADT_UNS_KEYS = [
    "adt_var_names",
    "protein_names",
    "ADT_names",
    "adt_names",
    "adt_features",
    "protein_features",
    "citeseq_names",
]
DEFAULT_CLUSTER_KEYS = ["cluster", "seurat_clusters", "cell_type", "louvain", "leiden"]


def pick_first(keys, available):
    for key in keys:
        if key in available:
            return key
    return None


def to_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(np.asarray(matrix))


def read_mtx(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return mmread(handle).tocsr()
    return mmread(path).tocsr()


def find_usa_dir(base_dir):
    base = Path(base_dir)
    if base.is_file():
        base = base.parent

    candidates = [("spliced.mtx", "unspliced.mtx"), ("spliced.mtx.gz", "unspliced.mtx.gz")]
    for spliced_name, unspliced_name in candidates:
        if (base / spliced_name).exists() and (base / unspliced_name).exists():
            return base

    for spliced_path in base.rglob("spliced.mtx*"):
        rel_depth = len(spliced_path.relative_to(base).parts)
        if rel_depth > 5:
            continue
        for suffix in [".mtx", ".mtx.gz"]:
            candidate = spliced_path.with_name("unspliced" + suffix)
            if candidate.exists():
                return spliced_path.parent
    return None


def find_metadata_file(usa_dir, names):
    for name in names:
        candidate = usa_dir / name
        if candidate.exists():
            return candidate
    return None


def load_usa_metadata(usa_dir, genes_file, barcodes_file):
    usa_dir = Path(usa_dir)
    genes_path = Path(genes_file) if genes_file else find_metadata_file(usa_dir, ["features.tsv", "genes.tsv", "genes.tsv.gz"]) 
    barcodes_path = Path(barcodes_file) if barcodes_file else find_metadata_file(usa_dir, ["barcodes.tsv", "barcodes.tsv.gz"]) 

    if genes_path is None or not genes_path.exists():
        raise ValueError("USA genes/features file not found. Provide --af-genes-file.")
    if barcodes_path is None or not barcodes_path.exists():
        raise ValueError("USA barcodes file not found. Provide --af-barcodes-file.")

    if str(genes_path).endswith(".gz"):
        genes_df = pd.read_csv(genes_path, sep="\t", header=None, compression="gzip")
    else:
        genes_df = pd.read_csv(genes_path, sep="\t", header=None)

    if str(barcodes_path).endswith(".gz"):
        barcodes = pd.read_csv(barcodes_path, sep="\t", header=None, compression="gzip")[0].astype(str).tolist()
    else:
        barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)[0].astype(str).tolist()

    return genes_df, barcodes


def choose_gene_labels(genes_df, target_gene_names):
    cols = [genes_df.iloc[:, i].astype(str).tolist() for i in range(genes_df.shape[1])]
    overlaps = [len(set(col) & set(target_gene_names)) for col in cols]
    best_idx = int(np.argmax(overlaps)) if overlaps else 0
    label = cols[best_idx]
    if genes_df.shape[1] > 1:
        print("[usa] gene column selected: %d (overlap=%d)" % (best_idx, overlaps[best_idx]))
    return label


def choose_barcodes(barcodes, target_barcodes):
    targets = set(target_barcodes)
    variants = {
        "raw": barcodes,
        "strip-1": [b[:-2] if b.endswith("-1") else b for b in barcodes],
        "add-1": [b + "-1" if not b.endswith("-1") else b for b in barcodes],
    }
    scores = {name: len(set(vals) & targets) for name, vals in variants.items()}
    best_name = max(scores, key=scores.get)
    print("[usa] barcode mode: %s (overlap=%d)" % (best_name, scores[best_name]))
    return variants[best_name]


def orient_usa_matrix(matrix, genes, barcodes, label):
    if matrix.shape == (len(genes), len(barcodes)):
        return matrix.T
    if matrix.shape == (len(barcodes), len(genes)):
        return matrix
    raise ValueError("%s matrix shape does not match genes/barcodes lists." % label)


def load_usa_matrices(usa_dir, genes_file, barcodes_file, adata, min_overlap):
    usa_dir = Path(usa_dir)
    usa_dir = find_usa_dir(usa_dir) or usa_dir

    spliced_path = find_metadata_file(usa_dir, ["spliced.mtx", "spliced.mtx.gz"]) 
    unspliced_path = find_metadata_file(usa_dir, ["unspliced.mtx", "unspliced.mtx.gz"]) 
    ambiguous_path = find_metadata_file(usa_dir, ["ambiguous.mtx", "ambiguous.mtx.gz"])
    if spliced_path is None or unspliced_path is None:
        raise ValueError("USA matrices not found in %s" % usa_dir)

    genes_df, barcodes = load_usa_metadata(usa_dir, genes_file, barcodes_file)
    genes = choose_gene_labels(genes_df, adata.var_names)
    barcodes = choose_barcodes(barcodes, adata.obs_names)

    spliced = orient_usa_matrix(read_mtx(spliced_path), genes, barcodes, "spliced")
    unspliced = orient_usa_matrix(read_mtx(unspliced_path), genes, barcodes, "unspliced")
    if ambiguous_path is not None:
        print("[usa] ambiguous matrix detected")
        ambiguous = orient_usa_matrix(read_mtx(ambiguous_path), genes, barcodes, "ambiguous")
    else:
        ambiguous = None

    obs_index = {name: i for i, name in enumerate(adata.obs_names)}
    var_index = {name: i for i, name in enumerate(adata.var_names)}

    barcode_idx = [i for i, bc in enumerate(barcodes) if bc in obs_index]
    gene_idx = [i for i, g in enumerate(genes) if g in var_index]

    if len(barcode_idx) < min_overlap:
        raise ValueError("Too few barcode matches (%d)." % len(barcode_idx))
    if len(gene_idx) < min_overlap:
        raise ValueError("Too few gene matches (%d)." % len(gene_idx))

    spliced_sub = spliced[barcode_idx][:, gene_idx].tocoo()
    unspliced_sub = unspliced[barcode_idx][:, gene_idx].tocoo()
    if ambiguous is not None:
        ambiguous_sub = ambiguous[barcode_idx][:, gene_idx].tocoo()
    else:
        ambiguous_sub = None

    obs_target = np.array([obs_index[barcodes[i]] for i in barcode_idx], dtype=np.int64)
    var_target = np.array([var_index[genes[i]] for i in gene_idx], dtype=np.int64)

    spliced_out = sparse.coo_matrix(
        (spliced_sub.data, (obs_target[spliced_sub.row], var_target[spliced_sub.col])),
        shape=(adata.n_obs, adata.n_vars),
    ).tocsr()
    unspliced_out = sparse.coo_matrix(
        (unspliced_sub.data, (obs_target[unspliced_sub.row], var_target[unspliced_sub.col])),
        shape=(adata.n_obs, adata.n_vars),
    ).tocsr()
    if ambiguous_sub is not None:
        ambiguous_out = sparse.coo_matrix(
            (ambiguous_sub.data, (obs_target[ambiguous_sub.row], var_target[ambiguous_sub.col])),
            shape=(adata.n_obs, adata.n_vars),
        ).tocsr()
    else:
        ambiguous_out = None

    print("[usa] matched barcodes: %d | matched genes: %d" % (len(barcode_idx), len(gene_idx)))
    return spliced_out.tocsr(), unspliced_out.tocsr(), ambiguous_out


def merge_loom_layers(adata, loom_path, overwrite_layers):
    try:
        import scvelo as scv
    except ImportError as exc:
        raise ImportError("scvelo is required to read loom files. Install scvelo or skip --loom.") from exc

    ldata = scv.read(loom_path, cache=False)

    if not adata.obs_names.is_unique:
        print("[warn] input obs_names are not unique; merging may drop duplicates")
    if not ldata.obs_names.is_unique:
        print("[warn] loom obs_names are not unique; merging may drop duplicates")

    common_cells = adata.obs_names.intersection(ldata.obs_names)
    if len(common_cells) == 0:
        raise ValueError("No overlapping cell barcodes between h5ad and loom.")

    adata = adata[common_cells].copy()
    ldata = ldata[common_cells].copy()

    common_genes = adata.var_names.intersection(ldata.var_names)
    if len(common_genes) == 0:
        raise ValueError("No overlapping genes between h5ad and loom.")

    adata = adata[:, common_genes].copy()
    ldata = ldata[:, common_genes].copy()

    for layer_key in ["spliced", "unspliced", "ambiguous"]:
        if layer_key in ldata.layers and (overwrite_layers or layer_key not in adata.layers):
            adata.layers[layer_key] = ldata.layers[layer_key]

    print(
        "[merge] cells: %d | genes: %d | layers added: %s"
        % (adata.n_obs, adata.n_vars, ", ".join([k for k in ["spliced", "unspliced", "ambiguous"] if k in adata.layers]))
    )
    return adata


def ensure_layers(
    adata,
    spliced_key,
    unspliced_key,
    use_x_as_spliced,
    overwrite_layers,
    af_usa_dir,
    af_genes_file,
    af_barcodes_file,
    af_run_cmd,
    af_force,
    af_min_overlap,
):
    layer_keys = list(adata.layers.keys())

    if spliced_key is None:
        spliced_key = pick_first(DEFAULT_SPLICED_KEYS, layer_keys)
    if unspliced_key is None:
        unspliced_key = pick_first(DEFAULT_UNSPLICED_KEYS, layer_keys)

    if spliced_key is None and use_x_as_spliced:
        adata.layers["spliced"] = adata.X
        spliced_key = "spliced"

    missing_layers = spliced_key is None or unspliced_key is None
    if (missing_layers or af_force) and (af_usa_dir or af_run_cmd):
        if af_run_cmd:
            subprocess.run(af_run_cmd, shell=True, check=True)
        if not af_usa_dir:
            raise ValueError("--af-usa-dir is required to import alevin-fry USA matrices.")

        spliced_usa, unspliced_usa, ambiguous_usa = load_usa_matrices(
            af_usa_dir, af_genes_file, af_barcodes_file, adata, af_min_overlap
        )
        adata.layers["spliced"] = spliced_usa
        adata.layers["unspliced"] = unspliced_usa
        if ambiguous_usa is not None and (overwrite_layers or "ambiguous" not in adata.layers):
            adata.layers["ambiguous"] = ambiguous_usa
        spliced_key = "spliced"
        unspliced_key = "unspliced"

    if spliced_key is None:
        raise ValueError(
            "No spliced layer found. Provide --spliced-key, --use-x-as-spliced, --af-usa-dir, or --loom."
        )
    if unspliced_key is None:
        raise ValueError("No unspliced layer found. Provide --unspliced-key, --af-usa-dir, or --loom.")

    if spliced_key != "spliced" and (overwrite_layers or "spliced" not in adata.layers):
        adata.layers["spliced"] = adata.layers[spliced_key]
    if unspliced_key != "unspliced" and (overwrite_layers or "unspliced" not in adata.layers):
        adata.layers["unspliced"] = adata.layers[unspliced_key]

    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise ValueError("Missing spliced/unspliced layers after mapping.")

    adata.layers["spliced"] = to_csr(adata.layers["spliced"])
    adata.layers["unspliced"] = to_csr(adata.layers["unspliced"])
    adata.X = adata.layers["spliced"]


def extract_adt_names(adata, adt_key, adt_names_key):
    if adt_names_key and adt_names_key in adata.uns:
        return [str(n) for n in adata.uns[adt_names_key]]

    if "adt_var_names" in adata.uns:
        return [str(n) for n in adata.uns["adt_var_names"]]

    if adt_key in adata.obsm and isinstance(adata.obsm[adt_key], pd.DataFrame):
        return [str(n) for n in adata.obsm[adt_key].columns.tolist()]

    for key in DEFAULT_ADT_UNS_KEYS:
        if key in adata.uns:
            return [str(n) for n in adata.uns[key]]

    return None


def ensure_adt(adata, adt_key, adt_names_key):
    if adt_key is None:
        adt_key = pick_first(DEFAULT_ADT_OBSM_KEYS, adata.obsm.keys())
    if adt_key is None:
        raise ValueError("No ADT matrix found in obsm. Provide --adt-key.")

    adt_matrix = adata.obsm[adt_key]
    adt_names = extract_adt_names(adata, adt_key, adt_names_key)
    if adt_names is None:
        raise ValueError("No ADT names found. Provide --adt-names-key or store names in obsm columns.")

    adata.obsm["X_adt"] = to_csr(adt_matrix)
    adata.uns["adt_var_names"] = adt_names

    if adata.obsm["X_adt"].shape[1] != len(adata.uns["adt_var_names"]):
        raise ValueError("ADT matrix column count does not match adt_var_names length.")


def ensure_cluster(adata, cluster_key, cluster_source):
    if cluster_key in adata.obs:
        return

    if cluster_source and cluster_source in adata.obs:
        adata.obs[cluster_key] = adata.obs[cluster_source].astype(str)
        return

    auto_key = pick_first(DEFAULT_CLUSTER_KEYS, adata.obs.columns)
    if auto_key:
        adata.obs[cluster_key] = adata.obs[auto_key].astype(str)
        return

    adata.obs[cluster_key] = "unknown"
    print("[warn] No cluster column found; created obs['%s']='unknown'" % cluster_key)


def main():
    parser = argparse.ArgumentParser(description="Convert h5ad to veloline-ready format.")
    parser.add_argument("--input", required=True, help="Path to input h5ad")
    parser.add_argument("--output", required=True, help="Path to output h5ad")
    parser.add_argument("--loom", default=None, help="Optional loom file to merge spliced/unspliced")
    parser.add_argument("--spliced-key", default=None, help="Layer key to use as spliced")
    parser.add_argument("--unspliced-key", default=None, help="Layer key to use as unspliced")
    parser.add_argument("--adt-key", default=None, help="obsm key containing ADT counts")
    parser.add_argument("--adt-names-key", default=None, help="uns key containing ADT names")
    parser.add_argument("--cluster-key", default="cluster", help="Target obs column for clusters")
    parser.add_argument("--cluster-source", default=None, help="Source obs column for clusters")
    parser.add_argument("--use-x-as-spliced", action="store_true", help="Use .X as spliced if layers missing")
    parser.add_argument("--overwrite-layers", action="store_true", help="Overwrite existing spliced/unspliced")
    parser.add_argument("--af-usa-dir", default=None, help="Alevin-fry USA directory with spliced/unspliced mtx")
    parser.add_argument("--af-genes-file", default=None, help="Override genes/features file path")
    parser.add_argument("--af-barcodes-file", default=None, help="Override barcodes file path")
    parser.add_argument("--af-run-cmd", default=None, help="Shell command to run alevin-fry before import")
    parser.add_argument("--af-force", action="store_true", help="Force alevin-fry import even if layers exist")
    parser.add_argument("--af-min-overlap", type=int, default=10, help="Min overlap for genes/barcodes")

    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    print("[input] cells: %d | genes: %d" % (adata.n_obs, adata.n_vars))

    if args.loom:
        adata = merge_loom_layers(adata, args.loom, args.overwrite_layers)

    ensure_layers(
        adata,
        args.spliced_key,
        args.unspliced_key,
        args.use_x_as_spliced,
        args.overwrite_layers,
        args.af_usa_dir,
        args.af_genes_file,
        args.af_barcodes_file,
        args.af_run_cmd,
        args.af_force,
        args.af_min_overlap,
    )
    ensure_adt(adata, args.adt_key, args.adt_names_key)
    ensure_cluster(adata, args.cluster_key, args.cluster_source)

    if not adata.obs_names.is_unique:
        print("[warn] obs_names were not unique; making unique")
        adata.obs_names_make_unique()
    if not adata.var_names.is_unique:
        print("[warn] var_names were not unique; making unique")
        adata.var_names_make_unique()

    adata.write_h5ad(args.output)
    print("[output] wrote %s" % args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("[error] %s" % exc)
        sys.exit(1)
