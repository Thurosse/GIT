# References - veloline-run-inference

Quick links for stage 2 (fit1/fit2) context and dependencies.

## Internal

- AGENTS.md - stage map, model variants, and run layout
- veloline/stages/inference.py - inference stage entrypoint
- veloline/fit1.py - fit1 runner
- veloline/fit2.py - fit2 runner
- veloline/models.py - Pyro model registry and variants
- veloline/metaparams.py - workflow gates and model names
- veloline/io_state.py - posteriors and RNG state I/O
- veloline/splines_torch_fixed.py - B-spline basis used by models

## External

- https://docs.pyro.ai/en/stable/inference.html - Pyro inference (SVI, MCMC)
- https://pyro.ai/examples/ - Pyro examples and patterns
- https://pytorch.org/docs/stable/ - tensor ops and autograd
- https://scvelo.readthedocs.io/ - RNA velocity context
