"""Section 8 — assemble the central `mp` MetaparContainer.

`mp` is the namedtuple consumed by every Pyro model and helper. This module
wraps cell 44's inline construction as a single `build_mp(adata, data_to_fit)`
call.

`MP_SCHEMA_VERSION` is bumped whenever fields are added or renamed; downstream
state files refuse to load if the version doesn't match.
"""

import copy
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import torch

from splines_torch_fixed import spline_prep, derivative

from veloline.data_loading import df_mapping
from veloline.metaparams import (
    PHI_MIN, PHI_MAX,
    NU_SPLINE_DF, NU_SPLINE_DEGREE,
    K_SPLINE_DF, K_SPLINE_DEGREE,
    OMEGA_SPLINE_DF, OMEGA_SPLINE_DEGREE,
    MU_NU_0, SIGMA_NU_0, MU_NU_I, SIGMA_NU_I,
    MU_OMEGA_0, SIGMA_OMEGA_0, MU_OMEGA_I, SIGMA_OMEGA_I,
    MU_OMEGA, SIGMA_OMEGA,
    MU_GAMMA, SIGMA_GAMMA, MU_BETA, SIGMA_BETA,
    MU_A, SIGMA_A, MU_B, SIGMA_B,
    GAMMA_ALPHA, GAMMA_BETA,
    FIT1_VARIABLE_MODE, FIT2_VARIABLE_MODE,
    FIT1_EXPLICIT_GENE_NAMES, FIT2_EXPLICIT_GENE_NAMES,
    FIT_1_MODEL, FIT_2_MODEL,
    FIT_1_INFERENCE_BACKEND, FIT_2_INFERENCE_BACKEND,
    FIT_1_RESULT_REDUCTION, FIT_1_TO_FIT_2_SOURCE, FIT_1_TO_FIT_2_CONSTANTS,
    USE_BATCHING,
    FIT_1_CELL_BATCH_SIZE, FIT_1_GENE_BATCH_SIZE,
    FIT_2_CELL_BATCH_SIZE, FIT_2_GENE_BATCH_SIZE,
    BASELINE_ENABLED, BASELINE_BRANCH, BASELINE_FAMILY,
    BASELINE_AMPLITUDE, BASELINE_DECAY_RATE, BASELINE_FLOOR,
    FIT_1_MCMC_NUM_SAMPLES, FIT_1_MCMC_WARMUP_STEPS,
    FIT_1_MCMC_TARGET_ACCEPT, FIT_1_MCMC_MAX_TREE_DEPTH,
    FIT_2_MCMC_NUM_SAMPLES, FIT_2_MCMC_WARMUP_STEPS,
    FIT_2_MCMC_TARGET_ACCEPT, FIT_2_MCMC_MAX_TREE_DEPTH,
    MODEL_WORKFLOW,
    device,
)


MP_SCHEMA_VERSION = 1


def _resolve_gene_indices(mode, explicit_names, all_gene_names, matched_idx):
    """Pick which RNA gene indices to use for a given stage."""
    all_idx = np.arange(len(all_gene_names), dtype=np.int64)
    matched_idx = np.asarray(matched_idx, dtype=np.int64)

    if mode == "all_rna":
        return all_idx
    if mode == "rna_adt_matched":
        return matched_idx
    if mode == "explicit_names":
        name_to_idx = {g: i for i, g in enumerate(all_gene_names)}
        explicit_idx = [name_to_idx[g] for g in explicit_names if g in name_to_idx]
        missing = [g for g in explicit_names if g not in name_to_idx]
        if missing:
            print(f"[selection-warning] {len(missing)} explicit genes were not found, e.g. {missing[:5]}")
        return np.asarray(explicit_idx, dtype=np.int64)
    raise ValueError(f"Unknown variable mode: {mode}")


def build_mp(adata, data_to_fit):
    """Build the central MetaparContainer `mp` consumed by all Pyro models.

    Args:
        adata: the original AnnData (provides `.uns["adt_var_names"]`).
        data_to_fit: the model-fitting AnnData built in §6.4 (gene-filtered).
    Returns:
        A `MetaparContainer` namedtuple with all derived tensors.
    """
    # ── ADT ↔ RNA matching ────────────────────────────────────────────────────
    gene_names_all = np.array(data_to_fit.var_names)
    adt_var_names = list(adata.uns["adt_var_names"])

    matched_gene_idx = []
    matched_protein_idx = []
    for gi, g in enumerate(gene_names_all):
        hits = df_mapping[df_mapping["RNA_Marker"] == g]["ADT_Protein"]
        if len(hits) == 0:
            continue
        prot = hits.iloc[0]
        if prot in adt_var_names:
            matched_gene_idx.append(gi)
            matched_protein_idx.append(adt_var_names.index(prot))

    matched_gene_idx = np.asarray(matched_gene_idx, dtype=np.int64)
    matched_protein_idx = np.asarray(matched_protein_idx, dtype=np.int64)

    fit1_gene_idx = _resolve_gene_indices(FIT1_VARIABLE_MODE, FIT1_EXPLICIT_GENE_NAMES, gene_names_all, matched_gene_idx)
    fit2_gene_idx_raw = _resolve_gene_indices(FIT2_VARIABLE_MODE, FIT2_EXPLICIT_GENE_NAMES, gene_names_all, matched_gene_idx)

    if len(fit1_gene_idx) == 0:
        raise ValueError("Fit1 variable selection returned 0 genes.")

    _gene_to_prot = {int(g): int(p) for g, p in zip(matched_gene_idx.tolist(), matched_protein_idx.tolist())}
    fit2_gene_idx = np.asarray([int(g) for g in fit2_gene_idx_raw if int(g) in _gene_to_prot], dtype=np.int64)
    fit2_protein_idx = np.asarray([_gene_to_prot[int(g)] for g in fit2_gene_idx], dtype=np.int64)

    if len(fit2_gene_idx) == 0:
        raise ValueError("Fit2 variable selection returned 0 RNA+ADT matched genes after alignment.")

    print(f"ADT↔RNA matched pairs available: {len(matched_gene_idx)}")
    print(f"Fit1 genes selected: {len(fit1_gene_idx)}")
    print(f"Fit2 genes selected (RNA+ADT aligned): {len(fit2_gene_idx)}")

    # ── Scalar config → mp base dict ──────────────────────────────────────────
    metapars = dict(
        ø=PHI_MIN,
        χ=PHI_MAX,

        spline_df=NU_SPLINE_DF,
        spline_degree=NU_SPLINE_DEGREE,
        spline_df_k=K_SPLINE_DF,
        spline_degree_k=K_SPLINE_DEGREE,

        μμν0=MU_NU_0,
        σμν0=SIGMA_NU_0,
        μμνi=MU_NU_I,
        σμνi=SIGMA_NU_I,

        ωspline_df=OMEGA_SPLINE_DF,
        ωspline_degree=OMEGA_SPLINE_DEGREE,
        μω0=MU_OMEGA_0,
        σω0=SIGMA_OMEGA_0,
        μωi=MU_OMEGA_I,
        σωi=SIGMA_OMEGA_I,

        μω=MU_OMEGA,
        σω=SIGMA_OMEGA,

        Ng=len(fit1_gene_idx),
        Nc=data_to_fit.n_obs,

        workflow=copy.deepcopy(MODEL_WORKFLOW),
        fit_1_model_name=FIT_1_MODEL,
        fit_2_model_name=FIT_2_MODEL,
        fit_1_inference_backend=FIT_1_INFERENCE_BACKEND,
        fit_2_inference_backend=FIT_2_INFERENCE_BACKEND,
        fit_1_result_reduction=FIT_1_RESULT_REDUCTION,
        fit_1_to_fit_2_source=FIT_1_TO_FIT_2_SOURCE,
        fit_1_to_fit_2_constants=list(FIT_1_TO_FIT_2_CONSTANTS),

        use_batching=USE_BATCHING,
        fit_1_cell_batch_size=FIT_1_CELL_BATCH_SIZE,
        fit_1_gene_batch_size=FIT_1_GENE_BATCH_SIZE,
        fit_2_cell_batch_size=FIT_2_CELL_BATCH_SIZE,
        fit_2_gene_batch_size=FIT_2_GENE_BATCH_SIZE,

        baseline_enabled=BASELINE_ENABLED,
        baseline_branch=BASELINE_BRANCH,
        baseline_family=BASELINE_FAMILY,
        baseline_amplitude=BASELINE_AMPLITUDE,
        baseline_decay_rate=BASELINE_DECAY_RATE,
        baseline_floor=BASELINE_FLOOR,

        fit_1_mcmc_num_samples=FIT_1_MCMC_NUM_SAMPLES,
        fit_1_mcmc_warmup_steps=FIT_1_MCMC_WARMUP_STEPS,
        fit_1_mcmc_target_accept=FIT_1_MCMC_TARGET_ACCEPT,
        fit_1_mcmc_max_tree_depth=FIT_1_MCMC_MAX_TREE_DEPTH,

        fit_2_mcmc_num_samples=FIT_2_MCMC_NUM_SAMPLES,
        fit_2_mcmc_warmup_steps=FIT_2_MCMC_WARMUP_STEPS,
        fit_2_mcmc_target_accept=FIT_2_MCMC_TARGET_ACCEPT,
        fit_2_mcmc_max_tree_depth=FIT_2_MCMC_MAX_TREE_DEPTH,

        device=device,
        gamma_alpha=torch.tensor(GAMMA_ALPHA).float(),
        gamma_beta=torch.tensor(GAMMA_BETA).float(),

        ϕ=None,  # placeholder — filled after fit1
    )

    MetaparContainer = namedtuple("MetaparContainer", list(metapars.keys()))
    mp = MetaparContainer(**metapars)

    # ── Spline preparation ────────────────────────────────────────────────────
    t, k = spline_prep(lower_bound=mp.ø, upper_bound=mp.χ, df=mp.spline_df, degree=mp.spline_degree)
    tder, c, kder = derivative(t, k)

    tp, kp = spline_prep(lower_bound=mp.ø, upper_bound=mp.χ, df=mp.spline_df_k, degree=mp.spline_degree_k)
    tderp, cp, kderp = derivative(tp, kp)
    tder2p, c2p, kder2p = derivative(tderp, kderp)
    tder3p, c3p, kder3p = derivative(tder2p, kder2p)

    tω, kω = spline_prep(lower_bound=mp.ø, upper_bound=mp.χ, df=mp.ωspline_df, degree=mp.ωspline_degree)

    Nh = len(t) - k
    Nhω = len(tω) - kω
    Nhp = len(tp) - kp

    # ── Prior mean/std tensors for spline weights ─────────────────────────────
    μν = torch.zeros(Nh)
    μν[0] = mp.μμν0
    μν[1:] = mp.μμνi
    σν = torch.zeros(Nh)
    σν[0] = mp.σμν0
    σν[1:] = mp.σμνi

    μνp = torch.zeros(Nhp)
    μνp[0] = mp.μμν0
    μνp[1:] = mp.μμνi
    σνp = torch.zeros(Nhp)
    σνp[0] = mp.σμν0
    σνp[1:] = mp.σμνi

    μνω = torch.Tensor([mp.μω0] + [mp.μωi] * (Nhω - 1))
    σνω = torch.Tensor([mp.σω0] + [mp.σωi] * (Nhω - 1))

    # ── Tensor construction (S, U, P) ─────────────────────────────────────────
    S_all = torch.tensor(data_to_fit.layers["spliced"].toarray().astype(np.int64)).T.float().to(device)
    U_all = torch.tensor(data_to_fit.layers["unspliced"].toarray().astype(np.int64)).T.float().to(device)

    fit1_gene_idx_t = torch.tensor(fit1_gene_idx, dtype=torch.long, device=device)
    fit2_gene_idx_t = torch.tensor(fit2_gene_idx, dtype=torch.long, device=device)

    S_fit1 = torch.index_select(S_all, 0, fit1_gene_idx_t)
    U_fit1 = torch.index_select(U_all, 0, fit1_gene_idx_t)
    S_fit2 = torch.index_select(S_all, 0, fit2_gene_idx_t)
    U_fit2 = torch.index_select(U_all, 0, fit2_gene_idx_t)

    X_adt_fit2 = data_to_fit.obsm["X_adt"][:, fit2_protein_idx]
    if sp.issparse(X_adt_fit2):
        X_adt_fit2 = X_adt_fit2.toarray()
    P_fit2 = torch.tensor(X_adt_fit2.astype(np.int64)).T.float().to(device)

    # ── Count factors (log library-size offsets) ──────────────────────────────
    s_sum = np.asarray(data_to_fit.layers["spliced"].sum(1)).squeeze()
    s_sum_clamped = np.where(s_sum == 0, 1.0, s_sum).astype(np.float32)
    S_UMI_per_cell = torch.tensor(s_sum_clamped).float().to(device)
    count_factor = torch.log(S_UMI_per_cell / torch.mean(S_UMI_per_cell))

    p_sum = np.asarray(data_to_fit.obsm["X_adt"].sum(1)).squeeze()
    p_sum_clamped = np.where(p_sum == 0, 1.0, p_sum).astype(np.float32)
    P_UMI_per_cell = torch.tensor(p_sum_clamped).float().to(device)
    count_factor_P = torch.log(P_UMI_per_cell / torch.mean(P_UMI_per_cell))

    # ── Gene-level prior tensors for kinetic rates ────────────────────────────
    def _repeat_scalar_prior(v, n):
        return torch.tensor(v).float().repeat([n, 1]).to(device)

    μγ_fit1 = _repeat_scalar_prior(MU_GAMMA, len(fit1_gene_idx))
    σγ_fit1 = _repeat_scalar_prior(SIGMA_GAMMA, len(fit1_gene_idx))
    μβ_fit1 = _repeat_scalar_prior(MU_BETA, len(fit1_gene_idx))
    σβ_fit1 = _repeat_scalar_prior(SIGMA_BETA, len(fit1_gene_idx))

    μγ_fit2 = _repeat_scalar_prior(MU_GAMMA, len(fit2_gene_idx))
    σγ_fit2 = _repeat_scalar_prior(SIGMA_GAMMA, len(fit2_gene_idx))
    μβ_fit2 = _repeat_scalar_prior(MU_BETA, len(fit2_gene_idx))
    σβ_fit2 = _repeat_scalar_prior(SIGMA_BETA, len(fit2_gene_idx))

    μA_fit2 = _repeat_scalar_prior(MU_A, len(fit2_gene_idx))
    σA_fit2 = _repeat_scalar_prior(SIGMA_A, len(fit2_gene_idx))
    μB_fit2 = _repeat_scalar_prior(MU_B, len(fit2_gene_idx))
    σB_fit2 = _repeat_scalar_prior(SIGMA_B, len(fit2_gene_idx))

    # ── Assemble precomputed values into mp ───────────────────────────────────
    precompvals = dict(
        S=S_fit1,
        U=U_fit1,
        P=P_fit2,
        Ng=len(fit1_gene_idx),
        Ng_p=len(fit2_gene_idx),
        μγ=μγ_fit1,
        σγ=σγ_fit1,
        μβ=μβ_fit1,
        σβ=σβ_fit1,
        μA=μA_fit2,
        σA=σA_fit2,
        μB=μB_fit2,
        σB=σB_fit2,

        S_fit1=S_fit1,
        U_fit1=U_fit1,
        Ng_fit1=len(fit1_gene_idx),
        μγ_fit1=μγ_fit1,
        σγ_fit1=σγ_fit1,
        μβ_fit1=μβ_fit1,
        σβ_fit1=σβ_fit1,
        fit1_gene_idx=torch.tensor(fit1_gene_idx, dtype=torch.long, device=device),
        fit1_gene_names=np.array(gene_names_all[fit1_gene_idx]),

        S_fit2=S_fit2,
        U_fit2=U_fit2,
        P_fit2=P_fit2,
        Ng_fit2=len(fit2_gene_idx),
        Ng_p_fit2=len(fit2_gene_idx),
        μγ_fit2=μγ_fit2,
        σγ_fit2=σγ_fit2,
        μβ_fit2=μβ_fit2,
        σβ_fit2=σβ_fit2,
        μA_fit2=μA_fit2,
        σA_fit2=σA_fit2,
        μB_fit2=μB_fit2,
        σB_fit2=σB_fit2,
        fit2_gene_idx=torch.tensor(fit2_gene_idx, dtype=torch.long, device=device),
        fit2_gene_names=np.array(gene_names_all[fit2_gene_idx]),
    )

    metapars.update(precompvals)
    metapars.update({
        "μν": μν, "σν": σν,
        "μνp": μνp, "σνp": σνp,
        "μνω": μνω, "σνω": σνω,
        "count_factor": count_factor,
        "count_factor_P": count_factor_P,
        "spline_t": t, "spline_k": k,
        "spline_tp": tp, "spline_kp": kp,
        "spline_tder": tder, "spline_tderp": tderp,
        "spline_tder2p": tder2p, "spline_tder3p": tder3p,
        "spline_c": c, "spline_cp": cp, "spline_c2p": c2p, "spline_c3p": c3p,
        "spline_kder": kder, "spline_kderp": kderp,
        "spline_kder2p": kder2p, "spline_kder3p": kder3p,
        "spline_tω": tω, "spline_kω": kω,
        "Nh": Nh, "Nhp": Nhp, "Nhω": Nhω,
        "gene_names_all": gene_names_all,
        "phi_const_fit2": None,
        "nu_const_fit2": None,
        "ElogS_const_fit2": None,
        "ElogU_const_fit2": None,
        "logβg_const_fit2": None,
        "logγg_const_fit2": None,
        "shape_inv_const_fit2": None,
    })
    MetaparContainer = namedtuple("MetaparContainer", list(metapars.keys()))
    mp = MetaparContainer(**metapars)
    print(f"mp built with {len(mp._fields)} fields, schema v{MP_SCHEMA_VERSION}")
    return mp
