# References - veloline-pipeline-setup

Quick links for stage 1 (setup) context and dependencies.

## Internal

- AGENTS.md - canonical stage map and run layout
- veloline/stages/setup.py - setup stage entrypoint
- veloline/metaparams.py - all tunable knobs and device flag
- veloline/data_loading.py - AnnData load and barcode demux
- veloline/preprocess.py - filtering, HVG, normalization, data_to_fit
- veloline/mp_builder.py - build mp container
- veloline/io_state.py - run dir, state save/load
- veloline_driver.ipynb - thin notebook driver

## External

- https://anndata.readthedocs.io/ - AnnData structure and I/O
- https://scanpy.readthedocs.io/ - preprocessing and normalization
- https://scvelo.readthedocs.io/ - spliced/unspliced conventions
- https://pytorch.org/docs/stable/ - tensor and device basics
