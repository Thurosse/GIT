"""Section 9 — FIT 1: infer pseudotime ϕ via SVI or NUTS, extract posteriors.

Drives cells 48 (workflow setup tail) → 49 (run) → 50 (extract).
"""

import torch
import pyro
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

from veloline.metaparams import (
    FIT_1_N_INITS, FIT_1_WARMUP, FIT_1_N_ITER, FIT_1_LR, FIT_1_BETAS,
    FIT_1_NUM_SAMPLES, INVERSE_DPT,
)
from veloline.utils import _clear_active_batch_subsamples, check_model
from veloline.models import (
    MODEL_REGISTRY,
    _resolve_model,
    _activate_stage_mp,
    _rebuild_mp,
    _reduce_samples,
    _run_backend,
)


def run_fit1(mp, fit1_cfg):
    """Run FIT 1 inference end-to-end and return (mp_fit1, res_fit1, pps_fit1, posteriors).

    `posteriors` is a dict with keys ϕ_fit, ν_fit1, ζ_fit1, ζ_dϕ_fit1 (if applicable),
    ElogS_fit1, ElogU_fit1 (if applicable), logγg_fit1, logβg_fit1, ElogS2_fit1, ElogU2_fit1.
    """
    fit1_model_fn = _resolve_model(fit1_cfg["model_name"])
    mp_fit1 = _activate_stage_mp(mp, "fit_1")

    if getattr(mp_fit1, "use_batching", False):
        fit1_model_full = fit1_model_fn
    else:
        condition_on_phase = {
            obs_name: getattr(mp_fit1, obs_name)
            for obs_name in fit1_cfg.get("observables", ["S", "U"])
            if hasattr(mp_fit1, obs_name)
        }
        fit1_model_full = pyro.condition(fit1_model_fn, data=condition_on_phase)
    check_model(fit1_model_full, mp_fit1)

    res_fit1 = _run_backend(
        fit1_model_full,
        mp_fit1,
        backend=fit1_cfg.get("inference_backend", "svi"),
        svi_cfg=dict(
            n_inits=FIT_1_N_INITS,
            warmup=FIT_1_WARMUP,
            n_iter=FIT_1_N_ITER,
            lr=FIT_1_LR,
            betas=FIT_1_BETAS,
        ),
        mcmc_cfg=fit1_cfg.get("mcmc", {}),
        stage="fit_1",
    )

    # ── Posterior predictive ──────────────────────────────────────────────────
    
    
    if getattr(mp_fit1, "use_batching", False):
        mp_fit1_predict = _rebuild_mp(
            mp_fit1,
            {
                "use_batching": False,
                "fit_1_cell_batch_size": -1,
                "fit_1_gene_batch_size": -1,
            },
        )
    else:
        mp_fit1_predict = _rebuild_mp(mp_fit1, {"use_batching": True})

    if res_fit1["backend"] == "svi":
        meanfield_fit1 = AutoDiagonalNormal(fit1_model_full)
        predictive_fit1 = Predictive(fit1_model_fn, guide=meanfield_fit1, num_samples=FIT_1_NUM_SAMPLES)
        pps_fit1 = predictive_fit1(mp_fit1_predict)
    else:
        posterior_samples = res_fit1["mcmc"].get_samples()
        predictive_fit1 = Predictive(fit1_model_fn, posterior_samples=posterior_samples, num_samples=FIT_1_NUM_SAMPLES)
        pps_fit1 = predictive_fit1(mp_fit1_predict) | posterior_samples

    # ── Extract fitted quantities ─────────────────────────────────────────────
    reduction_mode = fit1_cfg.get("result_reduction", "mean")
    posteriors = {}
    if fit1_cfg["model_name"] == "fit1_with_spliced_and_unspliced_model":
        ϕ_fit = _reduce_samples(pps_fit1["ϕ"], reduction_mode).squeeze().cpu()
        ν_fit1 = _reduce_samples(pps_fit1["ν"], reduction_mode).squeeze().cpu()
        ζ_fit1 = _reduce_samples(pps_fit1["ζ"], reduction_mode).squeeze().cpu()
        ζ_dϕ_fit1 = _reduce_samples(pps_fit1["ζ_dϕ"], reduction_mode).squeeze().cpu()
        ElogS_fit1 = _reduce_samples(pps_fit1["ElogS"], reduction_mode).squeeze().cpu()
        ElogU_fit1 = _reduce_samples(pps_fit1["ElogU"], reduction_mode).squeeze().cpu()
        logγg_fit1 = _reduce_samples(pps_fit1["logγg"], reduction_mode).squeeze().cpu()
        logβg_fit1 = _reduce_samples(pps_fit1["logβg"], reduction_mode).squeeze().cpu()
        avg_count_factor = torch.mean(mp_fit1_predict.count_factor).cpu()
        ElogS2_fit1 = torch.einsum("gh,ch->gc", ν_fit1, ζ_fit1).cpu() + avg_count_factor.cpu()
        ElogU2_fit1 = (
            -logβg_fit1[:, None]
            + torch.log(
                torch.relu(
                    torch.einsum("gh,ch->gc", ν_fit1, ζ_dϕ_fit1)
                    + torch.exp(logγg_fit1)[:, None]
                )
                + 1e-5
            )
            + ElogS2_fit1
        ).cpu()
        posteriors.update(dict(
            ϕ_fit=ϕ_fit, ν_fit1=ν_fit1, ζ_fit1=ζ_fit1, ζ_dϕ_fit1=ζ_dϕ_fit1,
            ElogS_fit1=ElogS_fit1, ElogU_fit1=ElogU_fit1,
            logγg_fit1=logγg_fit1, logβg_fit1=logβg_fit1,
            ElogS2_fit1=ElogS2_fit1, ElogU2_fit1=ElogU2_fit1,
        ))
    else:
        ϕ_fit = _reduce_samples(pps_fit1["ϕ"], reduction_mode).squeeze().cpu()
        ν_fit1 = _reduce_samples(pps_fit1["ν"], reduction_mode).squeeze().cpu()
        ζ_fit1 = _reduce_samples(pps_fit1["ζ"], reduction_mode).squeeze().cpu()
        ElogS_fit1 = _reduce_samples(pps_fit1["ElogS"], reduction_mode).squeeze().cpu()
        avg_count_factor = torch.mean(mp_fit1_predict.count_factor).cpu()
        ElogS2_fit1 = torch.einsum("gh,ch->gc", ν_fit1, ζ_fit1).cpu() + avg_count_factor.cpu()
        posteriors.update(dict(
            ϕ_fit=ϕ_fit, ν_fit1=ν_fit1, ζ_fit1=ζ_fit1,
            ElogS_fit1=ElogS_fit1, ElogS2_fit1=ElogS2_fit1,
        ))

    if INVERSE_DPT:
        posteriors["ϕ_fit"] = 10 - posteriors["ϕ_fit"]

    print(f"ϕ range: [{ϕ_fit .min():.3f}, {ϕ_fit .max():.3f}]")
    return mp_fit1, res_fit1, pps_fit1, posteriors
