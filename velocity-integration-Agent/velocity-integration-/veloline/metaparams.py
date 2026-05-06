"""Section 2 of the notebook — the SINGLE edit point for tunable knobs.

Every other module in this package reads from here. Edits made in this file
propagate to both the notebook (via `from veloline.metaparams import *`) and
the three stage skills.

Cell 5 (meta-parameters) and cell 7 (prior hyper-parameters) are kept together
to preserve the project's "one edit point" architectural rule.
"""

import torch

# ══════════════════════════════════════════════════════════════════════════════
#  META-PARAMETERS  —  edit here, nowhere else
# ══════════════════════════════════════════════════════════════════════════════

# ── hardware ──────────────────────────────────────────────────────────────────
USE_GPU = False


# ── data loading ──────────────────────────────────────────────────────────────
data_all_path = {
    "1":  "/home/arthur/cache/sample_filtered_feature_bc_matrix_240430_1.h5ad",  # !! barcoding
    "5":  "/home/arthur/cache/sample_filtered_feature_bc_matrix_250819_5.h5ad",  # inverse dpt
    "7":  "/home/arthur/cache/sample_filtered_feature_bc_matrix_250819_7.h5ad",
    "91": "/home/arthur/cache/sample_filtered_feature_bc_matrix_250825_9_Hg1.h5ad",
    "92": "/home/arthur/cache/sample_filtered_feature_bc_matrix_250825_9_Hg2.h5ad",
    "94": "/home/arthur/cache/sample_filtered_feature_bc_matrix_250825_9_Hg4.h5ad",
}

DATA_PATH = data_all_path["7"]

#  ── barcode ──────────────────────────────────────────────────────────────
BARCODE_NAMES = ["Hu.Hg1", "Hu.Hg2"]
BARCODE_SELECTED = None    # put None for no barcode #0 for first

# ── cell-type filtering (clusters to EXCLUDE) ────────────────────────────────
cell_id = "cluster_BMM"

EXCLUDED_CLUSTERS = [
    "QC_removed",
    "Late Erythroid",
    "CD4 Memory T",
    "CD8 Memory T",
    "Naive T",
    "Early Erythroid",
    "Cycling Progenitor",
    "NK",
]


# ── gene selection ────────────────────────────────────────────────────────────
N_HIGHLY_VARIABLE   = -1 # top HVGs kept by seurat_v3 ; -1 to keep everything
MIN_CELLS_FRACTION  = 0.05    # gene must be expressed in ≥ this fraction of cells
# set to -1 to effectively disable the spliced / unspliced mean filter:
MIN_SPLICED_MEAN    = -1
MIN_UNSPLICED_MEAN  = -1

# ── ADT retention controls (applied after gene filters) ───────────────────────
FORCE_INCLUDE_ADT_MAPPED_GENES = True  # if True, keep ADT-mapped RNA genes even when they fail filters above
FORCED_ADT_PROTEINS = []  # optional subset of ADT proteins to force-include; [] => all mapped ADTs available in the data

# ── variable selection and model workflow configuration ───────────────────────
FIT1_VARIABLE_MODE = "rna_adt_matched"       # "all_rna" | "rna_adt_matched" | "explicit_names"
FIT2_VARIABLE_MODE = "rna_adt_matched"  # "all_rna" | "rna_adt_matched" | "explicit_names"
FIT1_EXPLICIT_GENE_NAMES = []
FIT2_EXPLICIT_GENE_NAMES = []

FIT_1_MODEL = "fit1_with_spliced_and_unspliced_model"  # | "fit1_latent_variable_model"
FIT_2_MODEL = "fit2_latent_variable_model"             # | "fit2_latent_variable_model0" | "fit2_latent_integral_separate_model" | "fit2_latent_integral_model"

FIT_1_INFERENCE_BACKEND = "svi"   # "svi" | "mcmc"
FIT_2_INFERENCE_BACKEND = "svi"   # "svi" | "mcmc"
FIT_1_RESULT_REDUCTION  = "mean"  # "mean" | "median" | "sample"
FIT_1_TO_FIT_2_SOURCE   = "fit1_posterior_mean"  # "fit1_posterior_mean" | "fit1_posterior_median" | "fit1_single_sample" | "none"
FIT_1_TO_FIT_2_CONSTANTS = ["phi", "nu", "logbeta", "loggamma", "shape_inv"]
# accepted: any subset of ["phi","nu","spliced","unspliced","logbeta","loggamma","shape_inv"] or ["all"]
# don't use spliced and unspliced in general

# ── batching controls (convergence/stability) ─────────────────────────────────
USE_BATCHING = False
FIT_1_CELL_BATCH_SIZE = 512
FIT_1_GENE_BATCH_SIZE = -1
FIT_2_CELL_BATCH_SIZE = 512
FIT_2_GENE_BATCH_SIZE = -1

# ── decaying baseline controls ─────────────────────────────────────────────────
BASELINE_ENABLED    = False
BASELINE_BRANCH     = "all"          # "all" | "su" | "p"
BASELINE_FAMILY     = "exponential"  # "exponential" | "linear"
BASELINE_AMPLITUDE  = 0.0
BASELINE_DECAY_RATE = 1.0
BASELINE_FLOOR      = 0.0

# ── MCMC controls ─────────────────────────────────────────────────────────────
FIT_1_MCMC_NUM_SAMPLES    = 200
FIT_1_MCMC_WARMUP_STEPS   = 200
FIT_1_MCMC_TARGET_ACCEPT  = 0.8
FIT_1_MCMC_MAX_TREE_DEPTH = 8

FIT_2_MCMC_NUM_SAMPLES    = 200
FIT_2_MCMC_WARMUP_STEPS   = 200
FIT_2_MCMC_TARGET_ACCEPT  = 0.8
FIT_2_MCMC_MAX_TREE_DEPTH = 8

# ── B-spline for ν in phase (spliced / RNA model) ONLY for pseudotime ────────
NU_SPLINE_DF     = 5
NU_SPLINE_DEGREE = 3

# ── B-spline for P in fit2_latent_variable_model only ────────────────────────
K_SPLINE_DF     = 12
K_SPLINE_DEGREE = 3

# ── B-spline for ω (velocity scaling) — constant (=1) in current pipeline ────
OMEGA_SPLINE_DF     = 4
OMEGA_SPLINE_DEGREE = 3
MU_OMEGA_0 = 0.5; SIGMA_OMEGA_0 = 0.03
MU_OMEGA_I = 0.0; SIGMA_OMEGA_I = 0.2
MU_OMEGA   = 0.5; SIGMA_OMEGA   = 0.25

# ── pseudotime range ─────────────────────────────────────────────────────────
PHI_MIN = 0.0   # ø
PHI_MAX = 10.0  # χ


# ── optimisation: FIT 1 (ϕ) ──────────────────────────────────────────────────
FIT_1_N_ITER  = 2000
FIT_1_LR      = 0.01
FIT_1_BETAS   = (0.80, 0.99)
FIT_1_N_INITS = 5    # random restarts for seed selection
FIT_1_WARMUP  = 20   # warm-up steps per restart

# ── optimisation: FIT 2 ──────────────────────────────────────────────────────
FIT_2_N_ITER  = 4000
FIT_2_LR      = 0.01
FIT_2_BETAS   = (0.80, 0.99)
FIT_2_N_INITS = 5
FIT_2_WARMUP  = 20


# ── fitting order ────────────────────────────────────────────────────────────
# FIT 1 — infer ϕ
# FIT 2 — condition on ϕ
# Step 3 — posterior predictive draws → extract expectations
RUN_FIT_1 = True
RUN_FIT_2 = True

# ── if the pseudo time is in the wrong order ─────────────────────────────────
INVERSE_DPT = False

# ── posterior predictive samples ─────────────────────────────────────────────
FIT_1_NUM_SAMPLES = 200
FIT_2_NUM_SAMPLES = 200

# ── dictionary-driven workflow (single control surface) ──────────────────────
MODEL_WORKFLOW = {
    "fit_1": {
        "enabled": RUN_FIT_1,
        "model_name": FIT_1_MODEL,
        "inference_backend": FIT_1_INFERENCE_BACKEND,
        "result_reduction": FIT_1_RESULT_REDUCTION,
        "variable_mode": FIT1_VARIABLE_MODE,
        "observables": ["S", "U"],
        "mcmc": {
            "num_samples": FIT_1_MCMC_NUM_SAMPLES,
            "warmup_steps": FIT_1_MCMC_WARMUP_STEPS,
            "target_accept_prob": FIT_1_MCMC_TARGET_ACCEPT,
            "max_tree_depth": FIT_1_MCMC_MAX_TREE_DEPTH,
        },
    },
    "fit_2": {
        "enabled": RUN_FIT_2,
        "model_name": FIT_2_MODEL,
        "inference_backend": FIT_2_INFERENCE_BACKEND,
        "fit1_source": FIT_1_TO_FIT_2_SOURCE,
        "fit1_constants": FIT_1_TO_FIT_2_CONSTANTS,
        "variable_mode": FIT2_VARIABLE_MODE,
        "observables": ["S", "U", "P"],
        "mcmc": {
            "num_samples": FIT_2_MCMC_NUM_SAMPLES,
            "warmup_steps": FIT_2_MCMC_WARMUP_STEPS,
            "target_accept_prob": FIT_2_MCMC_TARGET_ACCEPT,
            "max_tree_depth": FIT_2_MCMC_MAX_TREE_DEPTH,
        },
    },
}

# ── shift-analysis smoothing ─────────────────────────────────────────────────
SHIFT_WINDOW_SIZE = 200   # cells window for windowed derivative

# ── neighbourhood / diffusion map ────────────────────────────────────────────
N_NEIGHBORS = 50
N_PCS       = 10
N_DIFFMAP   = 3


# ══════════════════════════════════════════════════════════════════════════════
#  PRIOR HYPER-PARAMETERS  (cell 7)
# ══════════════════════════════════════════════════════════════════════════════
# ν (RNA spline weights)
MU_NU_0 =  0.0;  SIGMA_NU_0 = 0.6
MU_NU_I = -0.2;  SIGMA_NU_I = 1.5

# dispersion (Gamma prior on shape_inv)
GAMMA_ALPHA = 1.0
GAMMA_BETA  = 2.0
# kinetic rates β, γ  (log-normal)
MU_BETA  = 2.0;  SIGMA_BETA  = 1.0
MU_GAMMA = 0.0;  SIGMA_GAMMA = 0.5
# ADT rates A, B  (same parameterisation)
MU_B = 2.0;  SIGMA_B = 1.0
MU_A = 0.0;  SIGMA_A = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Derived / computed once from the above (do NOT edit below this line)
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda:0") if (USE_GPU and torch.cuda.is_available()) else torch.device("cpu")
torch.set_default_device(device)
