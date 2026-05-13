# Loom generation notes

Veloline needs `spliced` and `unspliced` layers. If they are missing, generate a loom file with velocyto and merge it into the h5ad.

## Velocyto run10x (10x + cellranger outputs)
```bash
velocyto run10x -m repeat_mask.gtf /path/to/cellranger_out/ genes.gtf
```

## Velocyto run (custom BAM + barcodes)
```bash
velocyto run -b barcodes.tsv -o /path/to/out -m repeat_mask.gtf possorted_genome_bam.bam genes.gtf
```

## Helper script
You can build these commands with:
```bash
python .agents/skills/veloline-h5ad-prep/scripts/generate_loom.py --mode run10x --sample-dir /path/to/cellranger_out --gtf genes.gtf --mask repeat_mask.gtf
```

After you have a loom file, merge it with:
```bash
python .agents/skills/veloline-h5ad-prep/scripts/convert_h5ad_for_veloline.py --input input.h5ad --output output_veloline.h5ad --loom sample.loom
```
