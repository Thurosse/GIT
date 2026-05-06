"""Section 10 — FIT 2: condition on ϕ, infer kinetics β, γ and ADT rates A, B.

Drives cells 56 (transfer) → 57 (condition) → 58 (run) → 59 (predictive) → 61 (extract).
"""

import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

from veloline.metaparams import (
    FIT_2_N_INITS, FIT_2_WARMUP, FIT_2_N_ITER, FIT_2_LR, FIT_2_BETAS,
    FIT_2_NUM_SAMPLES,
    device,
)
from veloline.utils import _clear_active_batch_subsamples, check_model
from veloline.models import (
    _resolve_model,
    _activate_stage_mp,
    _rebuild_mp,
    _reduce_samples,
    _run_backend,
)


# ── fit1 → fit2 constant transfer helpers (cell 56) ──────────────────────────

def _read_fit1_constants(constants):
    """Parse FIT_1_TO_FIT_2_CONSTANTS into a clean list of strings."""
    if constants is None:
        constants = ["phi"]
    if isinstance(constants, str):
        constants = [constants]
    requested = [str(item).strip() for item in constants if str(item).strip() != ""]
    if len(requested) == 0:
        raise ValueError("No fit1 constants requested.")
    return requested


def _align_fit1_tensor_to_fit2_genes(phase_tensor, mp):
    """Re-index a phase-stage tensor (shape [Ng_fit1, ...]) to the fit2 gene subset."""
    if phase_tensor is None:
        return None
    if not torch.is_tensor(phase_tensor) or phase_tensor.ndim == 0:
        return phase_tensor

    phase_names = [str(g) for g in list(mp.fit1_gene_names)]
    velocity_names = [str(g) for g in list(mp.fit2_gene_names)]
    if phase_tensor.shape[0] != len(phase_names):
        return phase_tensor

    phase_lookup = {g: i for i, g in enumerate(phase_names)}
    missing = [g for g in velocity_names if g not in phase_lookup]
    if missing:
        raise ValueError(f"Cannot align fit1 constants to fit2 genes; missing {len(missing)} genes, e.g. {missing[:5]}")

    idx = torch.tensor([phase_lookup[g] for g in velocity_names], dtype=torch.long, device=device)
    return torch.index_select(phase_tensor, 0, idx)


def _reduce_fit1_outputs(pps_fit1, reduction_mode):
    """Aggregate posterior sample tensors from pps_fit1 across the leading sample dim."""
    reduced = {}
    if pps_fit1 is None:
        return reduced
    for k, v in pps_fit1.items():
        if not torch.is_tensor(v):
            continue
        if v.ndim == 0:
            reduced[k] = v
        else:
            reduced[k] = _reduce_samples(v, reduction_mode)
    return reduced


def _coerce_override_tensor(x, mp):
    """Cast a constant override tensor to the same dtype/device as mp.S_fit2."""
    if not torch.is_tensor(x):
        return x
    ref = mp.S_fit2 if hasattr(mp, "S_fit2") else mp.S
    return x.to(device=device, dtype=ref.dtype)


def _build_fit2_constant_overrides(mp, fit1_cfg, fit2_cfg, pps_fit1):
    """Build the fit2 constant-overrides dict to pass to `_activate_stage_mp`."""
    requested = _read_fit1_constants(fit2_cfg.get("fit1_constants", ["phi"]))
    if len(requested) == 0:
        raise ValueError("Phase-to-fit2 constants list is empty.")

    supported = {
        "phi": ("ϕ", "phi_const_fit2"),
        "nu": ("ν", "nu_const_fit2"),
        "spliced": ("ElogS", "ElogS_const_fit2"),
        "unspliced": ("ElogU", "ElogU_const_fit2"),
        "logbeta": ("logβg", "logβg_const_fit2"),
        "loggamma": ("logγg", "logγg_const_fit2"),
        "shape_inv": ("shape_inv", "shape_inv_const_fit2"),
    }

    unknown = [k for k in requested if k != "all" and k not in supported]
    if len(unknown) > 0:
        raise ValueError(
            f"Unsupported fit1 constants requested: {unknown}. Supported: {list(supported.keys()) + ['all']}"
        )

    if "all" in requested:
        requested = list(supported.keys())

    overrides = {
        "phi_const_fit2": None,
        "nu_const_fit2": None,
        "ElogS_const_fit2": None,
        "ElogU_const_fit2": None,
        "logβg_const_fit2": None,
        "logγg_const_fit2": None,
        "shape_inv_const_fit2": None,
    }

    reduced_phase = {}
    if fit1_cfg.get("enabled", False):
        reduced_phase = _reduce_fit1_outputs(pps_fit1, fit1_cfg.get("result_reduction", "mean"))

    for key, (phase_key, override_key) in supported.items():
        if key not in requested:
            continue
        if phase_key not in reduced_phase:
            continue
        aligned = _align_fit1_tensor_to_fit2_genes(reduced_phase[phase_key], mp)
        overrides[override_key] = _coerce_override_tensor(aligned, mp)

    return overrides, requested


# ── End-to-end FIT 2 driver ──────────────────────────────────────────────────

def run_fit2(mp, fit1_cfg, fit2_cfg, pps_fit1, ElogS2_fit1=None, ElogU2_fit1=None):
    """Run FIT 2 inference end-to-end.

    Returns (mp_fit2, res_fit2, pps_fit2, posteriors).
    """
    fit2_model_fn = _resolve_model(fit2_cfg["model_name"])

    fit2_overrides, selected = _build_fit2_constant_overrides(mp, fit1_cfg, fit2_cfg, pps_fit1)
    print(f"Velocity fixed constants from fit1: {selected}")

    mp = _activate_stage_mp(mp, "fit_2", fit2_constant_overrides=fit2_overrides)

    if getattr(mp, "use_batching", False):
        fit2_model_full = fit2_model_fn
    else:
        condition_on_velo = {
            obs_name: getattr(mp, obs_name)
            for obs_name in fit2_cfg.get("observables", ["S", "U", "P"])
            if hasattr(mp, obs_name)
        }
        fit2_model_full = poutine.condition(fit2_model_fn, data=condition_on_velo)
    check_model(fit2_model_full, mp)

    res_fit2 = _run_backend(
        fit2_model_full,
        mp,
        backend=fit2_cfg.get("inference_backend", "svi"),
        svi_cfg=dict(
            n_inits=FIT_2_N_INITS,
            warmup=FIT_2_WARMUP,
            n_iter=FIT_2_N_ITER,
            lr=FIT_2_LR,
            betas=FIT_2_BETAS,
        ),
        mcmc_cfg=fit2_cfg.get("mcmc", {}),
        stage="fit_2",
    )

    # ── Posterior predictive ──────────────────────────────────────────────────
    
    if getattr(mp, "use_batching", False):
        mp_fit2_predict = _rebuild_mp(
            mp,
            {
                "use_batching": False,
                "fit_1_cell_batch_size": -1,
                "fit_1_gene_batch_size": -1,
                "fit_2_cell_batch_size": -1,
                "fit_2_gene_batch_size": -1,
            },
        )
    else:
        mp_fit2_predict = _rebuild_mp(mp, {"use_batching": False})

    if res_fit2["backend"] == "svi":
        meanfield_fit2 = AutoDiagonalNormal(fit2_model_full)
        predictive_fit2 = Predictive(fit2_model_fn, guide=meanfield_fit2, num_samples=FIT_2_NUM_SAMPLES)
        pps_fit2 = predictive_fit2(mp_fit2_predict)
    else:
        posterior_samples = res_fit2["mcmc"].get_samples()
        predictive_fit2 = Predictive(fit2_model_fn, posterior_samples=posterior_samples, num_samples=FIT_2_NUM_SAMPLES)
        pps_fit2 = predictive_fit2(mp_fit2_predict) | posterior_samples

    posteriors = extract_fit2_posteriors(
        pps_fit2, mp_fit2_predict, mp, fit2_cfg,
        ElogS2_fit1=ElogS2_fit1, ElogU2_fit1=ElogU2_fit1,
    )
    return mp, res_fit2, pps_fit2, posteriors


def extract_fit2_posteriors(pps_fit2, active_mp, mp, fit2_cfg,
                            ElogS2_fit1=None, ElogU2_fit1=None):
    """Model-aware posterior reconstruction (cell 61).

    Branches on `fit2_cfg["model_name"]` because each model exposes a different
    set of Pyro sites and computes ElogS/ElogU/ElogP differently.
    """
    EPS = 1e-5
    _MISSING = object()
    model_name = str(fit2_cfg.get("model_name", ""))

    def _posterior_mean(keys, default=_MISSING):
        if isinstance(keys, str):
            keys = (keys,)
        for key in keys:
            if key in pps_fit2 and torch.is_tensor(pps_fit2[key]):
                return pps_fit2[key].mean(0).squeeze()
        if default is not _MISSING:
            return default
        raise KeyError(f"None of the posterior keys {keys} were found in pps_fit2.")

    ϕ_fit = _posterior_mean(("ϕ"))
    if ϕ_fit.ndim != 1:
        ϕ_fit = ϕ_fit.squeeze()
    if ϕ_fit.ndim != 1:
        raise ValueError(f"ϕ_fit must be 1D, got shape {tuple(ϕ_fit.shape)}")

    Ng = int(getattr(active_mp, "Ng", _posterior_mean(("ElogP",)).shape[0]))
    Nc = int(getattr(active_mp, "Nc", ϕ_fit.shape[0]))
    if ϕ_fit.shape[0] != Nc:
        raise ValueError(f"Mismatch Nc={Nc} versus len(ϕ_fit)={ϕ_fit.shape[0]}")

    def _to_gene_vector(x, name):
        if x is None:
            return None
        x = x.to(ϕ_fit.device, dtype=ϕ_fit.dtype).squeeze()
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1D after squeeze, got shape {tuple(x.shape)}")
        if x.shape[0] != Ng:
            raise ValueError(f"{name} length must be Ng={Ng}, got {x.shape[0]}")
        return x

    def _to_gc(x, name):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=ϕ_fit.device, dtype=ϕ_fit.dtype)
        else:
            x = x.to(ϕ_fit.device, dtype=ϕ_fit.dtype)
        while x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.ndim == 0:
            x = x.expand(Ng, Nc)
        elif x.ndim == 1:
            if x.shape[0] == Nc:
                x = x.unsqueeze(0).expand(Ng, -1)
            elif x.shape[0] == Ng:
                x = x.unsqueeze(1).expand(-1, Nc)
            else:
                raise ValueError(f"{name}: cannot expand shape {tuple(x.shape)} to (Ng, Nc)")
        elif x.ndim == 2:
            if x.shape == (Ng, Nc):
                pass
            elif x.shape == (1, Nc):
                x = x.expand(Ng, -1)
            elif x.shape == (Ng, 1):
                x = x.expand(-1, Nc)
            elif x.shape[1] == Nc:
                x_aligned = _align_fit1_tensor_to_fit2_genes(x, active_mp)
                if x_aligned.shape == (Ng, Nc):
                    x = x_aligned
                else:
                    raise ValueError(f"{name}: alignment produced shape {tuple(x_aligned.shape)}")
            else:
                raise ValueError(f"{name}: cannot expand 2D shape {tuple(x.shape)} to (Ng, Nc)")
        else:
            raise ValueError(f"{name}: expected <=2D tensor after squeeze, got shape {tuple(x.shape)}")
        if x.shape != (Ng, Nc):
            raise ValueError(f"{name}: expected shape {(Ng, Nc)}, got {tuple(x.shape)}")
        return x

    def _mp_tensor(name):
        v = getattr(active_mp, name, None)
        return v if torch.is_tensor(v) else None

    ζ_fit = _posterior_mean(("ζ", "zeta"))
    ζ_dϕ_fit = _posterior_mean(("ζ_dϕ", "zeta_dphi"), default=None)
    ζ_d2ϕ_fit = _posterior_mean(("ζ_dϕ2", "ζ_d2ϕ", "zeta_d2phi", "ζ_dϕ"), default=ζ_dϕ_fit)

    ν_fit = _posterior_mean(("ν", "nu"), default=None)
    k_fit = _posterior_mean(("k",), default=None)
    Kg_fit = _posterior_mean(("Kg",), default=None)
    I_fit = _posterior_mean(("I",), default=None)

    logAg_fit = _posterior_mean(("logAg", "log_Ag"), default=None)
    Ag_fit = _posterior_mean(("Ag",), default=(torch.exp(logAg_fit) if logAg_fit is not None else None))

    logBg_fit = _posterior_mean(("logBg", "log_Bg"), default=None)
    Bg_fit = _posterior_mean(("Bg",), default=(torch.exp(logBg_fit) if logBg_fit is not None else None))
    if logBg_fit is None and Bg_fit is not None:
        logBg_fit = torch.log(torch.clamp(Bg_fit, min=EPS))

    logγg_fit = _posterior_mean(("logγg", "log_gamma_g"), default=None)
    γg_fit = _posterior_mean(("γg", "gamma_g"), default=(torch.exp(logγg_fit) if logγg_fit is not None else None))
    if logγg_fit is None and γg_fit is not None:
        logγg_fit = torch.log(torch.clamp(γg_fit, min=EPS))

    logβg_fit = _posterior_mean(("logβg", "log_beta_g"), default=None)
    βg_fit = _posterior_mean(("βg", "beta_g"), default=(torch.exp(logβg_fit) if logβg_fit is not None else None))
    if logβg_fit is None and βg_fit is not None:
        logβg_fit = torch.log(torch.clamp(βg_fit, min=EPS))

    ElogP_fit = _posterior_mean(("ElogP",))
    ElogS_fit = _posterior_mean(("ElogS",), default=None)
    ElogU_fit = _posterior_mean(("ElogU",), default=None)

    count_factor_P_tensor = _mp_tensor("count_factor_P")
    if count_factor_P_tensor is None:
        avg_count_factor_P = torch.tensor(0.0, device=ϕ_fit.device, dtype=ϕ_fit.dtype)
    else:
        avg_count_factor_P = torch.mean(count_factor_P_tensor.to(ϕ_fit.device, dtype=ϕ_fit.dtype))

    if model_name == "fit2_latent_variable_model":
        ElogP2_fit = torch.einsum("gh,ch->gc", k_fit, ζ_fit) + avg_count_factor_P
        ElogS2_fit = (
            -_to_gene_vector(logBg_fit, "logBg_fit")[:, None]
            + torch.log(torch.relu(torch.einsum("gh,ch->gc", k_fit, ζ_dϕ_fit) + _to_gene_vector(Ag_fit, "Ag_fit")[:, None]) + EPS)
            + ElogP2_fit
        )
        ElogU2_fit = (
            -_to_gene_vector(logβg_fit, "logβg_fit")[:, None]
            + torch.log(torch.relu(torch.einsum("gh,ch->gc", ν_fit, ζ_d2ϕ_fit) + _to_gene_vector(γg_fit, "γg_fit")[:, None]) + EPS)
            + ElogS2_fit
        )
    elif model_name == "fit2_latent_variable_model0":
        ElogP2_fit = torch.einsum("gh,ch->gc", k_fit, ζ_fit) + avg_count_factor_P
        ElogS2_fit = (
            -_to_gene_vector(logBg_fit, "logBg_fit")[:, None]
            + torch.log(torch.relu(torch.einsum("gh,ch->gc", k_fit, ζ_dϕ_fit) + _to_gene_vector(Ag_fit, "Ag_fit")[:, None]) + EPS)
            + ElogP2_fit
        )
        ElogU2_fit = (
            -_to_gene_vector(logβg_fit, "logβg_fit")[:, None]
            + torch.log(torch.relu(torch.einsum("gh,ch->gc", ν_fit, ζ_dϕ_fit) + _to_gene_vector(γg_fit, "γg_fit")[:, None]) + EPS)
            + ElogS2_fit
        )
    elif model_name == "fit2_latent_integral_model":
        ElogS2_fit = torch.relu(torch.einsum("gh,ch->gc", ν_fit, ζ_fit))
        ElogU2_fit = (
            -_to_gene_vector(logβg_fit, "logβg_fit")[:, None]
            + torch.log(torch.relu(torch.einsum("gh,ch->gc", ν_fit, ζ_dϕ_fit) + _to_gene_vector(γg_fit, "γg_fit")[:, None]) + EPS)
            + ElogS2_fit
        )
        ElogP2_fit = -(Ag_fit[:, None] * ϕ_fit) + torch.log(torch.relu(Kg_fit[:, None] + Bg_fit[:, None] * I_fit) + EPS)
    elif model_name == "fit2_latent_integral_separate_model":
        ElogP2_fit = -(Ag_fit[:, None] * ϕ_fit) + torch.log(torch.relu(Kg_fit[:, None] + Bg_fit[:, None] * I_fit) + EPS)
        ElogS2_fit = ElogS2_fit1
        ElogU2_fit = ElogU2_fit1
        if ElogS2_fit is None:
            ElogS2_fit = _to_gc(_mp_tensor("ElogS_const_fit2"), "ElogS_const_fit2")
        if ElogU2_fit is None:
            ElogU2_fit = _to_gc(_mp_tensor("ElogU_const_fit2"), "ElogU_const_fit2")
        if ElogS2_fit is None or ElogU2_fit is None:
            raise KeyError("ElogS2/ElogU2 unavailable for fit2_latent_integral_separate_model.")
    else:
        raise ValueError(f"Unsupported fit2 model '{model_name}' for model-aware reconstruction.")

    if ElogS_fit is None:
        ElogS_fit = ElogS2_fit1
    if ElogU_fit is None:
        ElogU_fit = ElogU2_fit1

    return {
        "ϕ_fit": ϕ_fit,
        "ζ_fit": ζ_fit,
        "ζ_dϕ_fit": ζ_dϕ_fit,
        "ζ_d2ϕ_fit": ζ_d2ϕ_fit,
        "ν_fit": ν_fit,
        "k_fit": k_fit,
        "Kg_fit": Kg_fit,
        "Ag_fit": Ag_fit,
        "Bg_fit": Bg_fit,
        "γg_fit": γg_fit,
        "βg_fit": βg_fit,
        "ElogP_fit": ElogP_fit,
        "ElogS_fit": ElogS_fit,
        "ElogU_fit": ElogU_fit,
        "ElogP2_fit": ElogP2_fit,
        "ElogS2_fit": ElogS2_fit,
        "ElogU2_fit": ElogU2_fit,
    }
