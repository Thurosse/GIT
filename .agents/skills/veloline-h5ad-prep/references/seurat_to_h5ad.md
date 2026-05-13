# Seurat to h5ad

Use SeuratDisk to export a Seurat object to h5ad.

```r
library(Seurat)
library(SeuratDisk)

SaveH5Seurat(seurat_obj, filename = "sample.h5seurat", overwrite = TRUE)
Convert("sample.h5seurat", dest = "h5ad", overwrite = TRUE)
```

Notes:
- Make sure the Seurat object includes both RNA and ADT assays before export.
- The resulting h5ad may not include spliced/unspliced layers; generate a loom file and merge if needed.
- After conversion, run the veloline h5ad prep conversion script to map keys.
