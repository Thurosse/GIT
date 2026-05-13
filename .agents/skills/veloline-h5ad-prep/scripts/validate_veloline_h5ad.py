#!/usr/bin/env python
import argparse
import ast
import sys
from pathlib import Path

import anndata as ad
from scipy import sparse


REQUIRED_LAYERS = ["spliced", "unspliced"]


def check_required(adata, cluster_key, report_adt_map):
    errors = []
    warnings = []

    for key in REQUIRED_LAYERS:
        if key not in adata.layers:
            errors.append("Missing layer: %s" % key)
        else:
            if not sparse.issparse(adata.layers[key]):
                warnings.append("Layer '%s' is not sparse; .toarray() may fail" % key)

    if "X_adt" not in adata.obsm:
        errors.append("Missing obsm['X_adt']")
    else:
        adt = adata.obsm["X_adt"]
        if getattr(adt, "shape", None) is None:
            errors.append("obsm['X_adt'] has no shape")
        elif adt.shape[0] != adata.n_obs:
            errors.append("obsm['X_adt'] rows != n_obs")

    if "adt_var_names" not in adata.uns:
        errors.append("Missing uns['adt_var_names']")
    else:
        adt_names = list(adata.uns["adt_var_names"])
        if "X_adt" in adata.obsm:
            adt = adata.obsm["X_adt"]
            if adt.shape[1] != len(adt_names):
                errors.append("adt_var_names length does not match X_adt columns")

    if cluster_key not in adata.obs:
        errors.append("Missing obs['%s']" % cluster_key)

    if not adata.obs_names.is_unique:
        warnings.append("obs_names are not unique")
    if not adata.var_names.is_unique:
        warnings.append("var_names are not unique")

    if report_adt_map and "adt_var_names" in adata.uns:
        adt_rna_map = None
        try:
            from veloline.data_loading import adt_rna_map
        except Exception:
            repo_root = Path(__file__).resolve().parents[4]
            data_loading = repo_root / "veloline" / "data_loading.py"
            if data_loading.exists():
                try:
                    tree = ast.parse(data_loading.read_text(encoding="utf-8"))
                    for node in tree.body:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id == "adt_rna_map":
                                    adt_rna_map = ast.literal_eval(node.value)
                                    break
                        if adt_rna_map is not None:
                            break
                except Exception as exc:
                    warnings.append("ADT map parse failed: %s" % exc)
            else:
                warnings.append("ADT map source not found at veloline/data_loading.py")

        if adt_rna_map is not None:
            adt_names = list(adata.uns["adt_var_names"])
            mapped = [n for n in adt_names if n in adt_rna_map]
            unmapped = [n for n in adt_names if n not in adt_rna_map]
            print("[adt-map] %d mapped | %d unmapped" % (len(mapped), len(unmapped)))
            if len(unmapped) > 0:
                print("[adt-map] unmapped examples: %s" % ", ".join(unmapped[:10]))

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate veloline h5ad prerequisites.")
    parser.add_argument("--input", required=True, help="Path to h5ad")
    parser.add_argument("--cluster-key", default="cluster", help="Cluster column name to validate")
    parser.add_argument("--report-adt-map", action="store_true", help="Report overlap with adt_rna_map")

    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    errors, warnings = check_required(adata, args.cluster_key, args.report_adt_map)

    if warnings:
        print("[warn] " + " | ".join(warnings))
    if errors:
        print("[fail] " + " | ".join(errors))
        sys.exit(1)

    print("[ok] veloline h5ad checks passed")


if __name__ == "__main__":
    main()
