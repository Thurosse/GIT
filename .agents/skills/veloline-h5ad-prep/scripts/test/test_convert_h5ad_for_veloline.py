import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import convert_h5ad_for_veloline as conv


def make_adata(n_cells=3, n_genes=4, genes=None, barcodes=None):
    if genes is None:
        genes = ["GeneA", "GeneB", "GeneC", "GeneD"][:n_genes]
    if barcodes is None:
        barcodes = ["cell1", "cell2", "cell3"][:n_cells]

    X = sparse.csr_matrix(np.arange(n_cells * n_genes).reshape(n_cells, n_genes))
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    adata.obs_names = barcodes
    return adata


class TestConvertH5adForVeloline(unittest.TestCase):
    def test_ensure_layers_auto_detect(self):
        adata = make_adata()
        adata.layers["spliced_counts"] = sparse.csr_matrix(np.ones((adata.n_obs, adata.n_vars)))
        adata.layers["unspliced_counts"] = sparse.csr_matrix(np.ones((adata.n_obs, adata.n_vars)) * 2)

        conv.ensure_layers(
            adata,
            spliced_key=None,
            unspliced_key=None,
            use_x_as_spliced=False,
            overwrite_layers=False,
            af_usa_dir=None,
            af_genes_file=None,
            af_barcodes_file=None,
            af_run_cmd=None,
            af_force=False,
            af_min_overlap=1,
        )

        self.assertIn("spliced", adata.layers)
        self.assertIn("unspliced", adata.layers)
        self.assertTrue(sparse.issparse(adata.layers["spliced"]))
        self.assertTrue(sparse.issparse(adata.layers["unspliced"]))

    def test_use_x_as_spliced(self):
        adata = make_adata()
        adata.layers["unspliced"] = sparse.csr_matrix(np.ones((adata.n_obs, adata.n_vars)))

        conv.ensure_layers(
            adata,
            spliced_key=None,
            unspliced_key=None,
            use_x_as_spliced=True,
            overwrite_layers=False,
            af_usa_dir=None,
            af_genes_file=None,
            af_barcodes_file=None,
            af_run_cmd=None,
            af_force=False,
            af_min_overlap=1,
        )

        self.assertIn("spliced", adata.layers)
        self.assertTrue(sparse.issparse(adata.layers["spliced"]))

    def test_ensure_adt_from_obsm_dataframe(self):
        adata = make_adata()
        adt_df = pd.DataFrame(
            [[1, 2], [3, 4], [5, 6]],
            index=adata.obs_names,
            columns=["ProtA", "ProtB"],
        )
        adata.obsm["protein_expression"] = adt_df

        conv.ensure_adt(adata, adt_key=None, adt_names_key=None)

        self.assertIn("X_adt", adata.obsm)
        self.assertIn("adt_var_names", adata.uns)
        self.assertEqual(len(adata.uns["adt_var_names"]), adata.obsm["X_adt"].shape[1])

    def test_ensure_cluster_auto(self):
        adata = make_adata()
        adata.obs["seurat_clusters"] = ["A", "B", "C"]

        conv.ensure_cluster(adata, cluster_key="cluster", cluster_source=None)

        self.assertIn("cluster", adata.obs)
        self.assertEqual(list(adata.obs["cluster"]), ["A", "B", "C"])

    def test_missing_layers_raises(self):
        adata = make_adata()

        with self.assertRaises(ValueError):
            conv.ensure_layers(
                adata,
                spliced_key=None,
                unspliced_key=None,
                use_x_as_spliced=False,
                overwrite_layers=False,
                af_usa_dir=None,
                af_genes_file=None,
                af_barcodes_file=None,
                af_run_cmd=None,
                af_force=False,
                af_min_overlap=1,
            )

    def test_af_usa_import(self):
        adata = make_adata()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            genes = ["gene1", "gene2", "gene3", "gene4"]
            gene_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
            features_path = tmp_path / "features.tsv"
            pd.DataFrame({0: genes, 1: gene_names}).to_csv(features_path, sep="\t", header=False, index=False)

            barcodes = ["cell1-1", "cell2-1", "cell3-1"]
            barcodes_path = tmp_path / "barcodes.tsv"
            pd.Series(barcodes).to_csv(barcodes_path, sep="\t", header=False, index=False)

            spliced = sparse.csr_matrix(np.ones((adata.n_obs, adata.n_vars)))
            unspliced = sparse.csr_matrix(np.ones((adata.n_obs, adata.n_vars)) * 3)
            ambiguous = sparse.csr_matrix(np.ones((adata.n_obs, adata.n_vars)) * 5)
            mmwrite(tmp_path / "spliced.mtx", spliced)
            mmwrite(tmp_path / "unspliced.mtx", unspliced)
            mmwrite(tmp_path / "ambiguous.mtx", ambiguous)

            conv.ensure_layers(
                adata,
                spliced_key=None,
                unspliced_key=None,
                use_x_as_spliced=False,
                overwrite_layers=False,
                af_usa_dir=str(tmp_path),
                af_genes_file=None,
                af_barcodes_file=None,
                af_run_cmd=None,
                af_force=True,
                af_min_overlap=1,
            )

        self.assertIn("spliced", adata.layers)
        self.assertIn("unspliced", adata.layers)
        self.assertIn("ambiguous", adata.layers)
        self.assertTrue(sparse.issparse(adata.layers["ambiguous"]))


if __name__ == "__main__":
    unittest.main()
