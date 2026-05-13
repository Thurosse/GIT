---
name: veloline-h5ad-prep
description: 'Convert and validate h5ad inputs for the veloline RNA-ADT velocity pipeline. Use when you need to fix spliced/unspliced layers, add X_adt/adt_var_names, merge or generate loom, or prep Seurat-derived data for veloline setup.'
argument-hint: 'Input h5ad path, optional loom path, and output path.'
---

# Veloline h5ad prep

Prepare any h5ad so it meets the veloline pipeline prerequisites:
- `.X` is spliced counts
- `.layers["spliced"]` and `.layers["unspliced"]`
- `.obsm["X_adt"]`
- `.uns["adt_var_names"]`
- `.obs["cluster"]` (or the cluster column you choose)

## When to use
- Convert h5ad for veloline setup or analysis
- Fix non-standard spliced/unspliced layer names
- Merge a loom file into an h5ad
- Convert Seurat-derived data into a veloline-ready h5ad

## Prerequisites
- Python env with `anndata`, `numpy`, `pandas`, `scipy`
- If importing USA matrices: alevin-fry outputs available on disk
- If running alevin-fry: `alevin-fry` installed and accessible on PATH
- If merging loom: `scvelo` installed
- If generating loom: `velocyto` installed and reference GTF + mask file
- If you need cellranger outputs, run in a Linux/WSL environment

## Workflow
1. **Seurat to h5ad (optional)**: If input is Seurat, convert first.
   - See [Seurat conversion](./references/seurat_to_h5ad.md).
2. **Alevin-fry (optional)**: If spliced/unspliced is missing, import USA matrices from alevin-fry.
   - See [Alevin-fry USA import](./references/alevin_fry_usa_import.md).
   - If spliced/unspliced are missing and you provide `--af-run-cmd`, the script runs alevin-fry first and then imports from `--af-usa-dir`.
   - If `ambiguous.mtx` is present, it is imported as `layers["ambiguous"]`.
3. **Generate loom (optional)**: If you prefer velocyto, create a loom file.
   - Use [generate_loom.py](./scripts/generate_loom.py) or the notes in [loom generation](./references/loom_generation.md).
4. **Convert / normalize keys**: Run the conversion script to fix keys and merge loom when needed.
   ```bash
   python .agents/skills/veloline-h5ad-prep/scripts/convert_h5ad_for_veloline.py \
     --input path/to/input.h5ad \
     --output path/to/output_veloline.h5ad \
     --loom path/to/sample.loom
   ```
    Example with alevin-fry USA import:
    ```bash
    python .agents/skills/veloline-h5ad-prep/scripts/convert_h5ad_for_veloline.py \
       --input input.h5ad \
       --output output_veloline.h5ad \
       --af-usa-dir /path/to/alevin_fry/usa
    ```
   Example with non-standard keys:
   ```bash
   python .agents/skills/veloline-h5ad-prep/scripts/convert_h5ad_for_veloline.py \
     --input input.h5ad \
     --output output_veloline.h5ad \
     --spliced-key spliced_counts \
     --unspliced-key unspliced_counts \
     --adt-key protein_expression \
     --adt-names-key protein_names \
     --cluster-source seurat_clusters
   ```
5. **Validate**: Confirm the required keys are present.
   ```bash
   python .agents/skills/veloline-h5ad-prep/scripts/validate_veloline_h5ad.py \
     --input path/to/output_veloline.h5ad \
     --cluster-key cluster \
     --report-adt-map
   ```
6. **Use in the pipeline**: Point `DATA_PATH` in [veloline/metaparams.py](veloline/metaparams.py) to the new h5ad.

## Key options
- `--spliced-key`, `--unspliced-key` to rename layers
- `--adt-key`, `--adt-names-key` to map ADT matrices and names
- `--cluster-key`, `--cluster-source` to standardize the cluster column
- `--use-x-as-spliced` if the h5ad only has `.X` counts
- `--overwrite-layers` to replace existing spliced/unspliced layers
- `--af-usa-dir` to import alevin-fry USA matrices (spliced/unspliced + optional ambiguous)
- `--af-run-cmd` to run alevin-fry before import when layers are missing

## Outputs
- A veloline-ready h5ad with required keys and standardized names

## References
- [Seurat conversion](./references/seurat_to_h5ad.md)
- [Loom generation notes](./references/loom_generation.md)
- [Alevin-fry USA import](./references/alevin_fry_usa_import.md)
