# Alevin-fry USA import

This skill can import spliced/unspliced matrices produced by alevin-fry.

## Expected files
Provide a directory that contains these files (optionally gzipped):
- `spliced.mtx` and `unspliced.mtx`
- `ambiguous.mtx` (optional)
- `barcodes.tsv`
- `features.tsv` or `genes.tsv`

The script will auto-discover these files under `--af-usa-dir`.
You can override file paths with `--af-genes-file` and `--af-barcodes-file`.

## Example
```bash
python .agents/skills/veloline-h5ad-prep/scripts/convert_h5ad_for_veloline.py \
  --input input.h5ad \
  --output output_veloline.h5ad \
  --af-usa-dir /path/to/alevin_fry/usa
```

## Optional: run alevin-fry first
If you want the skill to run alevin-fry before importing, pass a shell command:
```bash
python .agents/skills/veloline-h5ad-prep/scripts/convert_h5ad_for_veloline.py \
  --input input.h5ad \
  --output output_veloline.h5ad \
  --af-run-cmd "<your alevin-fry pipeline command>" \
  --af-usa-dir /path/to/alevin_fry/usa
```

## Notes
- The script aligns barcodes and genes to the h5ad and fills missing entries with zeros.
- It auto-picks the features column that best matches the h5ad gene names.
- If overlap is too small, increase `--af-min-overlap` or check your gene identifiers.
- If `ambiguous.mtx` is present, it is imported as `layers["ambiguous"]`.
- When spliced/unspliced are missing and `--af-run-cmd` is provided, the converter runs alevin-fry before importing from `--af-usa-dir`.
