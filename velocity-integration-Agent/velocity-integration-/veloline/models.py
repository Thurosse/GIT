"""Section 7 — the six Pyro probabilistic models + MODEL_REGISTRY.

Contents:
  fit1_latent_variable_model              (notebook cell 32, function half)
  fit1_with_spliced_and_unspliced_model   (cell 34)
  fit2_latent_variable_model              (cell 36)
  fit2_latent_variable_model0             (cell 38)
  fit2_latent_integral_model              (cell 40)
  fit2_latent_integral_separate_model     (cell 42)
  MODEL_REGISTRY + helpers                (cell 48 head)
"""

from collections import namedtuple
import copy

import torch
import pyro
import pyro.distributions as dist

from splines_torch_fixed import torch_spline_basis  # device-aware B-spline basis

from veloline.utils import (
    _get_batch_sizes,
    _get_active_batch_subsample,
    _index_first,
    _index_last,
    _select_obs,
    _decaying_baseline,
)


# ══════════════════════════════════════════════════════════════════════════════
#  fit1 — pseudotime ϕ inference
# ══════════════════════════════════════════════════════════════════════════════

def fit1_latent_variable_model(mp):
    """Fit1 model: infers per-cell pseudotime ϕ and per-gene RNA spline weights ν.

    Observed: spliced counts S only.
    """
    device = mp.device
    cell_bs, gene_bs = _get_batch_sizes(mp, "fit_1")
    _stage = "fit_1"

    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device, subsample=_get_active_batch_subsample(_stage, "cells"))
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device, subsample=_get_active_batch_subsample(_stage, "genes"))

    with gene_plate as gene_idx:
        gene_idx = gene_idx.to(device)
        ν = pyro.sample("ν", dist.Normal(mp.μν.to(device), mp.σν.to(device)).to_event(1))

    with cell_plate as cell_idx:
        cell_idx = cell_idx.to(device)
        if mp.ϕ is None:
            ϕ = pyro.sample("ϕ", dist.Uniform(mp.ø, mp.χ))
        else:
            ϕ = mp.ϕ if cell_idx is None else torch.index_select(mp.ϕ, 0, cell_idx)
            pyro.deterministic("ϕ", ϕ)

        ζ = torch_spline_basis(ϕ.squeeze(), mp.spline_t, mp.spline_k, prepend=1)
        pyro.deterministic("ζ", ζ)

    count_factor = _index_last(mp.count_factor.to(device), cell_idx)
    baseline_su = _decaying_baseline(mp, ϕ, "su")
    ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + count_factor + baseline_su
    pyro.deterministic("ElogS", ElogS)

    with gene_plate:
        gene_idx = gene_idx.to(device)
        shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))

    with cell_plate, gene_plate:
        cell_idx = cell_idx.to(device)
        gene_idx = gene_idx.to(device)
        S_obs = _select_obs(mp.S.to(device), gene_idx, cell_idx)
        pyro.sample(
            "S",
            dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))),
            obs=S_obs,
        )


def fit1_with_spliced_and_unspliced_model(mp):
    """Fit1 model jointly explaining spliced S and unspliced U counts.

    ElogU = -logβg + log(ν·ζ_dϕ + γg) + ElogS  — the RNA velocity kinetic relation.
    """
    device = mp.device
    cell_bs, gene_bs = _get_batch_sizes(mp, "fit_1")
    _stage = "fit_1"

    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device, subsample=_get_active_batch_subsample(_stage, "cells"))
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device, subsample=_get_active_batch_subsample(_stage, "genes"))

    with cell_plate as cell_idx:
        cell_idx = cell_idx.to(device)
        if mp.ϕ is None:
            ϕ = pyro.sample("ϕ", dist.Uniform(mp.ø, mp.χ))
        else:
            ϕ = mp.ϕ if cell_idx is None else torch.index_select(mp.ϕ, 0, cell_idx)
            pyro.deterministic("ϕ", ϕ)

        ζ = torch_spline_basis(ϕ.squeeze(), mp.spline_t, mp.spline_k, prepend=1).to(device)
        pyro.deterministic("ζ", ζ)

        ζ_dϕ = torch_spline_basis(
            ϕ.squeeze(),
            mp.spline_tder,
            mp.spline_kder,
            mp.spline_c,
            prepend=0,
        ).to(device)
        pyro.deterministic("ζ_dϕ", ζ_dϕ)

    with gene_plate as gene_idx:
        gene_idx = gene_idx.to(device)
        shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        logγg = pyro.sample("logγg", dist.Normal(_index_first(mp.μγ.to(device), gene_idx), _index_first(mp.σγ.to(device), gene_idx)))
        logβg = pyro.sample("logβg", dist.Normal(_index_first(mp.μβ.to(device), gene_idx), _index_first(mp.σβ.to(device), gene_idx)))
        γg = torch.exp(logγg)
        pyro.deterministic("γg", γg)
        ν = pyro.sample("ν", dist.Normal(mp.μν.to(device), mp.σν.to(device)).to_event(1))

    count_factor = _index_last(mp.count_factor.to(device), cell_idx)
    baseline_su = _decaying_baseline(mp, ϕ, "su")

    ElogS = torch.einsum("...gch,ch->gc", ν, ζ) + count_factor + baseline_su
    pyro.deterministic("ElogS", ElogS)

    ElogU = -logβg + torch.log(
        torch.relu(torch.einsum("...gch,...ch->gc", ν, ζ_dϕ) + γg) + 1e-5
    ) + ElogS
    pyro.deterministic("ElogU", ElogU)

    with cell_plate, gene_plate:
        gene_idx = gene_idx.to(device)
        cell_idx = cell_idx.to(device)  # original notebook contains this assignment
        S_obs = _select_obs(mp.S.to(device), gene_idx, cell_idx)
        U_obs = _select_obs(mp.U.to(device), gene_idx, cell_idx)
        pyro.sample(
            "S",
            dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))),
            obs=S_obs,
        )
        pyro.sample(
            "U",
            dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogU))),
            obs=U_obs,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  fit2 — kinetics β, γ, A, B
# ══════════════════════════════════════════════════════════════════════════════

def fit2_latent_variable_model(mp):
    """fit2 model with two independent derivative spline sets for RNA and ADT branches."""
    device = mp.device
    cell_bs, gene_bs = _get_batch_sizes(mp, "fit_2")
    _stage = "fit_2"

    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device, subsample=_get_active_batch_subsample(_stage, "cells"))
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device, subsample=_get_active_batch_subsample(_stage, "genes"))

    with cell_plate as cell_idx:
        cell_idx = cell_idx.to(device)
        if getattr(mp, "phi_const_fit2", None) is not None:
            ϕ = _index_last(mp.phi_const_fit2.to(device), cell_idx)
            pyro.deterministic("ϕ", ϕ)
        else:
            ϕ = pyro.sample("ϕ", dist.Uniform(mp.ø, mp.χ))

        ζ = torch_spline_basis(ϕ.squeeze(), mp.spline_tp, mp.spline_kp, prepend=1)
        pyro.deterministic("ζ", ζ)
        ζ_dϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tderp, mp.spline_kderp, mp.spline_cp, prepend=0)
        pyro.deterministic("ζ_dϕ", ζ_dϕ)
        ζ_dϕ2 = torch_spline_basis(ϕ.squeeze(), mp.spline_tder, mp.spline_kder, mp.spline_c, prepend=0)
        pyro.deterministic("ζ_dϕ2", ζ_dϕ2)

    with gene_plate as gene_idx:
        gene_idx = gene_idx.to(device)
        if getattr(mp, "shape_inv_const_fit2", None) is not None:
            shape_inv = _index_first(mp.shape_inv_const_fit2.to(device), gene_idx)
            pyro.deterministic("shape_inv", shape_inv)
        else:
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        shape_inv_P = pyro.sample("shape_inv_P", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        if getattr(mp, "logγg_const_fit2", None) is not None:
            logγg = _index_first(mp.logγg_const_fit2.to(device), gene_idx)
            pyro.deterministic("logγg", logγg)
        else:
            logγg = pyro.sample("logγg", dist.Normal(_index_first(mp.μγ.to(device), gene_idx), _index_first(mp.σγ.to(device), gene_idx)))
        if getattr(mp, "logβg_const_fit2", None) is not None:
            logβg = _index_first(mp.logβg_const_fit2.to(device), gene_idx)
            pyro.deterministic("logβg", logβg)
        else:
            logβg = pyro.sample("logβg", dist.Normal(_index_first(mp.μβ.to(device), gene_idx), _index_first(mp.σβ.to(device), gene_idx)))
        γg = torch.exp(logγg)
        pyro.deterministic("γg", γg)
        if getattr(mp, "nu_const_fit2", None) is not None:
            ν = _index_first(mp.nu_const_fit2.to(device), gene_idx)
            pyro.deterministic("ν", ν)
        else:
            ν = pyro.sample("ν", dist.Normal(mp.μνp.to(device), mp.σνp.to(device)).to_event(1))

        k = pyro.sample("k", dist.Normal(mp.μνp.to(device), mp.σνp.to(device)).to_event(1))
        logAg = pyro.sample("logAg", dist.Normal(_index_first(mp.μA.to(device), gene_idx), _index_first(mp.σA.to(device), gene_idx)))
        logBg = pyro.sample("logBg", dist.Normal(_index_first(mp.μB.to(device), gene_idx), _index_first(mp.σB.to(device), gene_idx)))
        Ag = torch.exp(logAg)
        pyro.deterministic("Ag", Ag)

    count_factor = _index_last(mp.count_factor.to(device), cell_idx)
    avg_count_factor_P = torch.mean(mp.count_factor_P.to(device))
    baseline_p = _decaying_baseline(mp, ϕ, "p")
    baseline_su = _decaying_baseline(mp, ϕ, "su")

    ElogP = torch.einsum("gch,ch->gc", k, ζ) + avg_count_factor_P + baseline_p
    pyro.deterministic("ElogP", ElogP)

    if getattr(mp, "ElogS_const_fit2", None) is not None:
        ElogS = _select_obs(mp.ElogS_const_fit2.squeeze().to(device), gene_idx, cell_idx)
    else:
        ElogS = (-logBg + torch.log(torch.relu(torch.einsum("gch,ch->gc", k, ζ_dϕ) + Ag) + 1e-5) + ElogP + baseline_su)
    pyro.deterministic("ElogS", ElogS)

    if getattr(mp, "ElogU_const_fit2", None) is not None:
        ElogU = _select_obs(mp.ElogU_const_fit2.squeeze().to(device), gene_idx, cell_idx)
    else:
        ElogU = (-logβg + torch.log(torch.relu(torch.einsum("gch,ch->gc", ν, ζ_dϕ2) + γg) + 1e-5) + ElogS)
    pyro.deterministic("ElogU", ElogU)

    with gene_plate, cell_plate:
        cell_idx = cell_idx.to(device)
        gene_idx = gene_idx.to(device)
        S_obs = _select_obs(mp.S.to(device), gene_idx, cell_idx)
        U_obs = _select_obs(mp.U.to(device), gene_idx, cell_idx)
        P_obs = _select_obs(mp.P.to(device), gene_idx, cell_idx)
        pyro.sample("S", dist.GammaPoisson(1./shape_inv, 1./(shape_inv * torch.exp(ElogS))), obs=S_obs)
        pyro.sample("U", dist.GammaPoisson(1./shape_inv, 1./(shape_inv * torch.exp(ElogU))), obs=U_obs)
        pyro.sample("P", dist.GammaPoisson(1./shape_inv_P, 1./(shape_inv_P * torch.exp(ElogP))), obs=P_obs)


def fit2_latent_variable_model0(mp):
    """Legacy single-spline fit2 model (S, U, P) — same ζ_dϕ shared by RNA and ADT branches."""
    device = mp.device
    cell_bs, gene_bs = _get_batch_sizes(mp, "fit_2")
    _stage = "fit_2"

    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device, subsample=_get_active_batch_subsample(_stage, "cells"))
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device, subsample=_get_active_batch_subsample(_stage, "genes"))

    with cell_plate as cell_idx:
        cell_idx = cell_idx.to(device)
        if getattr(mp, "phi_const_fit2", None) is not None:
            ϕ = _index_last(mp.phi_const_fit2.to(device), cell_idx)
            pyro.deterministic("ϕ", ϕ)
        else:
            ϕ = pyro.sample("ϕ", dist.Uniform(mp.ø, mp.χ))

        ζ = torch_spline_basis(ϕ.squeeze(), mp.spline_tp, mp.spline_kp, prepend=1)
        pyro.deterministic("ζ", ζ)
        ζ_dϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tderp, mp.spline_kderp, mp.spline_cp, prepend=0)
        pyro.deterministic("ζ_dϕ", ζ_dϕ)

    with gene_plate as gene_idx:
        gene_idx = gene_idx.to(device)
        if getattr(mp, "shape_inv_const_fit2", None) is not None:
            shape_inv = _index_first(mp.shape_inv_const_fit2.to(device), gene_idx)
            pyro.deterministic("shape_inv", shape_inv)
        else:
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        shape_inv_P = pyro.sample("shape_inv_P", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))
        if getattr(mp, "logγg_const_fit2", None) is not None:
            logγg = _index_first(mp.logγg_const_fit2.to(device), gene_idx)
            pyro.deterministic("logγg", logγg)
        else:
            logγg = pyro.sample("logγg", dist.Normal(_index_first(mp.μγ.to(device), gene_idx), _index_first(mp.σγ.to(device), gene_idx)))
        if getattr(mp, "logβg_const_fit2", None) is not None:
            logβg = _index_first(mp.logβg_const_fit2.to(device), gene_idx)
            pyro.deterministic("logβg", logβg)
        else:
            logβg = pyro.sample("logβg", dist.Normal(_index_first(mp.μβ.to(device), gene_idx), _index_first(mp.σβ.to(device), gene_idx)))
        γg = torch.exp(logγg)
        pyro.deterministic("γg", γg)
        if getattr(mp, "nu_const_fit2", None) is not None:
            ν = _index_first(mp.nu_const_fit2.to(device), gene_idx)
            pyro.deterministic("ν", ν)
        else:
            ν = pyro.sample("ν", dist.Normal(mp.μνp.to(device), mp.σνp.to(device)).to_event(1))

        k = pyro.sample("k", dist.Normal(mp.μνp.to(device), mp.σνp.to(device)).to_event(1))
        logAg = pyro.sample("logAg", dist.Normal(_index_first(mp.μA.to(device), gene_idx), _index_first(mp.σA.to(device), gene_idx)))
        logBg = pyro.sample("logBg", dist.Normal(_index_first(mp.μB.to(device), gene_idx), _index_first(mp.σB.to(device), gene_idx)))
        Ag = torch.exp(logAg)
        pyro.deterministic("Ag", Ag)

    count_factor = _index_last(mp.count_factor.to(device), cell_idx)
    avg_count_factor_P = torch.mean(mp.count_factor_P.to(device))
    baseline_p = _decaying_baseline(mp, ϕ, "p")
    baseline_su = _decaying_baseline(mp, ϕ, "su")

    ElogP = torch.einsum("gch,ch->gc", k, ζ) + avg_count_factor_P + baseline_p
    pyro.deterministic("ElogP", ElogP)

    if getattr(mp, "ElogS_const_fit2", None) is not None:
        ElogS = _select_obs(mp.ElogS_const_fit2.squeeze().to(device), gene_idx, cell_idx)
    else:
        ElogS = (-logBg + torch.log(torch.relu(torch.einsum("gch,ch->gc", k, ζ_dϕ) + Ag) + 1e-5) + ElogP + baseline_su)
    pyro.deterministic("ElogS", ElogS)

    if getattr(mp, "ElogU_const_fit2", None) is not None:
        ElogU = _select_obs(mp.ElogU_const_fit2.squeeze().to(device), gene_idx, cell_idx)
    else:
        ElogU = (-logβg + torch.log(torch.relu(torch.einsum("gch,ch->gc", ν, ζ_dϕ) + γg) + 1e-5) + ElogS)
    pyro.deterministic("ElogU", ElogU)

    with gene_plate, cell_plate:
        cell_idx = cell_idx.to(device)
        gene_idx = gene_idx.to(device)
        S_obs = _select_obs(mp.S.to(device), gene_idx, cell_idx)
        U_obs = _select_obs(mp.U.to(device), gene_idx, cell_idx)
        P_obs = _select_obs(mp.P.to(device), gene_idx, cell_idx)
        pyro.sample("S", dist.GammaPoisson(1./shape_inv, 1./(shape_inv * torch.exp(ElogS))), obs=S_obs)
        pyro.sample("U", dist.GammaPoisson(1./shape_inv, 1./(shape_inv * torch.exp(ElogU))), obs=U_obs)
        pyro.sample("P", dist.GammaPoisson(1./shape_inv_P, 1./(shape_inv_P * torch.exp(ElogP))), obs=P_obs)


def _prep_const_gene_factory(device, Ng):
    """Coerce a fit1-stage constant to device and align to fit2 gene axis if needed."""
    def _prep_const_gene(x):
        if not torch.is_tensor(x):
            return x
        x = x.to(device)
        while x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x
    return _prep_const_gene


def fit2_latent_integral_model(mp):
    """Integral fit2 model: protein P modelled via analytic integral of the RNA spline trajectory."""
    device = mp.device
    cell_bs, gene_bs = _get_batch_sizes(mp, "fit_2")
    _stage = "fit_2"

    _prep_const_gene = _prep_const_gene_factory(device, mp.Ng)

    def _prep_const_obs(x):
        x = _prep_const_gene(x)
        if torch.is_tensor(x) and x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x

    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device, subsample=_get_active_batch_subsample(_stage, "cells"))
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device, subsample=_get_active_batch_subsample(_stage, "genes"))

    with cell_plate as cell_idx:
        if getattr(mp, "phi_const_fit2", None) is not None:
            ϕ = _index_last(mp.phi_const_fit2.to(device), cell_idx)
            pyro.deterministic("ϕ", ϕ)
        else:
            ϕ = pyro.sample("ϕ", dist.Uniform(mp.ø, mp.χ))

        ζ = torch_spline_basis(ϕ.squeeze(), mp.spline_tp, mp.spline_kp, prepend=1)
        pyro.deterministic("ζ", ζ)
        ζ_dϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tderp, mp.spline_kderp, mp.spline_cp, prepend=0)
        pyro.deterministic("ζ_dϕ", ζ_dϕ)
        ζ_d2ϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tder2p, mp.spline_kder2p, mp.spline_c2p, prepend=0)
        ζ_d3ϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tder3p, mp.spline_kder3p, mp.spline_c3p, prepend=0)

        h_target = ζ.shape[1]
        if ζ_d2ϕ.shape[1] < h_target:
            n_pad = h_target - ζ_d2ϕ.shape[1]
            pad_zeros = torch.zeros(ζ_d2ϕ.shape[0], n_pad, dtype=ζ_d2ϕ.dtype, device=device)
            ζ_d2ϕ = torch.cat([pad_zeros, ζ_d2ϕ], dim=1)
        pyro.deterministic("ζ_d2ϕ", ζ_d2ϕ)
        if ζ_d3ϕ.shape[1] < h_target:
            n_pad = h_target - ζ_d3ϕ.shape[1]
            pad_zeros = torch.zeros(ζ_d3ϕ.shape[0], n_pad, dtype=ζ_d3ϕ.dtype, device=device)
            ζ_d3ϕ = torch.cat([pad_zeros, ζ_d3ϕ], dim=1)
        pyro.deterministic("ζ_d3ϕ", ζ_d3ϕ)

    with gene_plate as gene_idx:
        if getattr(mp, "shape_inv_const_fit2", None) is not None:
            shape_const = _prep_const_gene(mp.shape_inv_const_fit2)
            shape_inv = _index_first(shape_const, gene_idx)
            pyro.deterministic("shape_inv", shape_inv)
        else:
            shape_inv = pyro.sample("shape_inv", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))

        shape_inv_P = pyro.sample("shape_inv_P", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))

        if getattr(mp, "logγg_const_fit2", None) is not None:
            logγ_const = _prep_const_gene(mp.logγg_const_fit2)
            logγg = _index_first(logγ_const, gene_idx)
            pyro.deterministic("logγg", logγg)
        else:
            logγg = pyro.sample("logγg", dist.Normal(_index_first(mp.μγ.to(device), gene_idx), _index_first(mp.σγ.to(device), gene_idx)))

        if getattr(mp, "logβg_const_fit2", None) is not None:
            logβ_const = _prep_const_gene(mp.logβg_const_fit2)
            logβg = _index_first(logβ_const, gene_idx)
            pyro.deterministic("logβg", logβg)
        else:
            logβg = pyro.sample("logβg", dist.Normal(_index_first(mp.μβ.to(device), gene_idx), _index_first(mp.σβ.to(device), gene_idx)))

        γg = torch.exp(logγg)
        pyro.deterministic("γg", γg)

        if getattr(mp, "nu_const_fit2", None) is not None:
            nu_const = _prep_const_gene(mp.nu_const_fit2)
            ν = _index_first(nu_const, gene_idx)
            pyro.deterministic("ν", ν)
        else:
            ν = pyro.sample("ν", dist.Normal(mp.μνp.to(device), mp.σνp.to(device)).to_event(1))

        Kg = pyro.sample("Kg", dist.Normal(mp.μB.to(device), mp.σB.to(device)))
        logAg = pyro.sample("logAg", dist.Normal(_index_first(mp.μA.to(device), gene_idx), _index_first(mp.σA.to(device), gene_idx)))
        logBg = pyro.sample("logBg", dist.Normal(_index_first(mp.μB.to(device), gene_idx), _index_first(mp.σB.to(device), gene_idx)))
        Ag = torch.exp(logAg)
        Bg = torch.exp(logBg)
        pyro.deterministic("Ag", Ag)
        pyro.deterministic("Bg", Bg)

    if getattr(mp, "ElogS_const_fit2", None) is not None:
        ElogS_const = _prep_const_obs(mp.ElogS_const_fit2)
        ElogS = _select_obs(ElogS_const.squeeze(), gene_idx, cell_idx)
    else:
        ElogS = torch.relu(torch.einsum("gch,ch->gc", ν, ζ))
    pyro.deterministic("ElogS", ElogS)

    if getattr(mp, "ElogU_const_fit2", None) is not None:
        ElogU_const = _prep_const_obs(mp.ElogU_const_fit2)
        ElogU = _select_obs(ElogU_const.squeeze(), gene_idx, cell_idx)
    else:
        ElogU = -logβg + torch.log(torch.relu(torch.einsum("gch,ch->gc", ν, ζ_dϕ) + γg) + 1e-5) + ElogS
    pyro.deterministic("ElogU", ElogU)

    exp_term = torch.exp(Ag * ϕ)
    smallest_phi_idx = torch.argsort(ϕ)[0]
    Jf = (
        (1.0 / (Ag + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ) - ζ[smallest_phi_idx, :])
        - (1.0 / (Ag ** 2 + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ_dϕ) + ζ_dϕ[smallest_phi_idx, :])
        + (1.0 / (Ag ** 3 + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ_d2ϕ) - ζ_d2ϕ[smallest_phi_idx, :])
        + (1.0 / (Ag ** 4 + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ_d3ϕ) - ζ_d3ϕ[smallest_phi_idx, :])
    )
    pyro.deterministic("Jf", Jf)

    I = torch.einsum("gch,gch->gc", ν, Jf)
    pyro.deterministic("I", I)

    ElogP = -(Ag * ϕ) + torch.log(torch.relu(Kg + Bg * I) + 1e-5)
    pyro.deterministic("ElogP", ElogP)

    with gene_plate, cell_plate:
        S_obs = _select_obs(mp.S.to(device), gene_idx, cell_idx)
        U_obs = _select_obs(mp.U.to(device), gene_idx, cell_idx)
        P_obs = _select_obs(mp.P.to(device), gene_idx, cell_idx)
        pyro.sample("S", dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogS))), obs=S_obs)
        pyro.sample("U", dist.GammaPoisson(1.0 / shape_inv, 1.0 / (shape_inv * torch.exp(ElogU))), obs=U_obs)
        pyro.sample("P", dist.GammaPoisson(1.0 / shape_inv_P, 1.0 / (shape_inv_P * torch.exp(ElogP))), obs=P_obs)


def fit2_latent_integral_separate_model(mp):
    """ADT-focused integral fit2 model: only protein P is observed/modelled.

    The cell_plate context is nested twice in the original code; preserved as-is.
    """
    device = mp.device
    cell_bs, gene_bs = _get_batch_sizes(mp, "fit_2")
    _stage = "fit_2"

    _prep_const_gene = _prep_const_gene_factory(device, mp.Ng)

    def _prep_const_obs(x):
        x = _prep_const_gene(x)
        if torch.is_tensor(x) and x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x

    cell_plate = pyro.plate("cells", mp.Nc, dim=-1, device=device, subsample=_get_active_batch_subsample(_stage, "cells"))
    gene_plate = pyro.plate("genes", mp.Ng, dim=-2, device=device, subsample=_get_active_batch_subsample(_stage, "genes"))

    with cell_plate as cell_idx:
        with cell_plate as cell_idx:
            cell_idx = cell_idx.to(device)
            if getattr(mp, "phi_const_fit2", None) is not None:
                ϕ = _index_last(mp.phi_const_fit2.to(device), cell_idx)
                pyro.deterministic("ϕ", ϕ)
            else:
                ϕ = pyro.sample("ϕ", dist.Uniform(mp.ø, mp.χ))

        ζ = torch_spline_basis(ϕ.squeeze(), mp.spline_tp, mp.spline_kp, prepend=1)
        pyro.deterministic("ζ", ζ)
        ζ_dϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tderp, mp.spline_kderp, mp.spline_cp, prepend=0)
        pyro.deterministic("ζ_dϕ", ζ_dϕ)
        ζ_d2ϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tder2p, mp.spline_kder2p, mp.spline_c2p, prepend=0)
        ζ_d3ϕ = torch_spline_basis(ϕ.squeeze(), mp.spline_tder3p, mp.spline_kder3p, mp.spline_c3p, prepend=0)

        h_target = ζ.shape[1]
        if ζ_d2ϕ.shape[1] < h_target:
            n_pad = h_target - ζ_d2ϕ.shape[1]
            pad_zeros = torch.zeros(ζ_d2ϕ.shape[0], n_pad, dtype=ζ_d2ϕ.dtype, device=device)
            ζ_d2ϕ = torch.cat([pad_zeros, ζ_d2ϕ], dim=1)
        pyro.deterministic("ζ_d2ϕ", ζ_d2ϕ)
        if ζ_d3ϕ.shape[1] < h_target:
            n_pad = h_target - ζ_d3ϕ.shape[1]
            pad_zeros = torch.zeros(ζ_d3ϕ.shape[0], n_pad, dtype=ζ_d3ϕ.dtype, device=device)
            ζ_d3ϕ = torch.cat([pad_zeros, ζ_d3ϕ], dim=1)
        pyro.deterministic("ζ_d3ϕ", ζ_d3ϕ)

    with gene_plate as gene_idx:
        gene_idx = gene_idx.to(device)
        if getattr(mp, "shape_inv_const_fit2", None) is not None:
            shape_const = _prep_const_gene(mp.shape_inv_const_fit2)
            shape_inv_P = _index_first(shape_const, gene_idx)
            pyro.deterministic("shape_inv_P", shape_inv_P)
        else:
            shape_inv_P = pyro.sample("shape_inv_P", dist.Gamma(mp.gamma_alpha.to(device), mp.gamma_beta.to(device)))

        if getattr(mp, "nu_const_fit2", None) is not None:
            nu_const = _prep_const_gene(mp.nu_const_fit2)
            ν = _index_first(nu_const, gene_idx)
            pyro.deterministic("ν", ν)
        else:
            ν = pyro.sample("ν", dist.Normal(mp.μB.to(device), mp.σB.to(device)).to_event(1))

        Kg = pyro.sample("Kg", dist.Normal(mp.μνp.to(device), mp.σνp.to(device)))
        logAg = pyro.sample("logAg", dist.Normal(_index_first(mp.μA.to(device), gene_idx), _index_first(mp.σA.to(device), gene_idx)))
        logBg = pyro.sample("logBg", dist.Normal(_index_first(mp.μB.to(device), gene_idx), _index_first(mp.σB.to(device), gene_idx)))
        Ag = torch.exp(logAg)
        Bg = torch.exp(logBg)
        pyro.deterministic("Ag", Ag)
        pyro.deterministic("Bg", Bg)

    if getattr(mp, "ElogS_const_fit2", None) is not None:
        ElogS_const = _prep_const_obs(mp.ElogS_const_fit2)
        ElogS = _select_obs(ElogS_const.squeeze(), gene_idx, cell_idx)
        pyro.deterministic("ElogS", ElogS)

    if getattr(mp, "ElogU_const_fit2", None) is not None:
        ElogU_const = _prep_const_obs(mp.ElogU_const_fit2)
        ElogU = _select_obs(ElogU_const.squeeze(), gene_idx, cell_idx)
        pyro.deterministic("ElogU", ElogU)

    exp_term = torch.exp(Ag * ϕ)
    smallest_phi_idx = torch.argsort(ϕ)[0]
    Jf = (
        (1.0 / (Ag + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ) - ζ[smallest_phi_idx, :])
        - (1.0 / (Ag ** 2 + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ_dϕ) + ζ_dϕ[smallest_phi_idx, :])
        + (1.0 / (Ag ** 3 + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ_d2ϕ) - ζ_d2ϕ[smallest_phi_idx, :])
        + (1.0 / (Ag ** 4 + 1e-5))[:, :, None] * (torch.einsum("gc,ch->gch", exp_term, ζ_d3ϕ) - ζ_d3ϕ[smallest_phi_idx, :])
    )
    pyro.deterministic("Jf", Jf)

    I = torch.einsum("gch,gch->gc", ν, Jf)
    pyro.deterministic("I", I)

    ElogP = -(Ag * ϕ) + torch.log(torch.relu(Kg + Bg * I) + 1e-5)
    pyro.deterministic("ElogP", ElogP)

    with gene_plate, cell_plate:
        cell_idx = cell_idx.to(device)
        gene_idx = gene_idx.to(device)
        P_obs = _select_obs(mp.P.to(device), gene_idx, cell_idx)
        rate = torch.clamp(1.0 / (shape_inv_P * torch.exp(ElogP)), min=1e-8)
        pyro.sample("P", dist.GammaPoisson(1.0 / shape_inv_P, rate), obs=P_obs)


# ══════════════════════════════════════════════════════════════════════════════
#  Registry + workflow helpers (cell 48 head)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "fit1_latent_variable_model": fit1_latent_variable_model,
    "fit1_with_spliced_and_unspliced_model": fit1_with_spliced_and_unspliced_model,
    "fit2_latent_variable_model": fit2_latent_variable_model,
    "fit2_latent_variable_model0": fit2_latent_variable_model0,
    "fit2_latent_integral_model": fit2_latent_integral_model,
    "fit2_latent_integral_separate_model": fit2_latent_integral_separate_model,
}


def _resolve_model(model_name):
    """Look up a model callable from the registry by name string."""
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model in workflow registry: {model_name}")
    return MODEL_REGISTRY[model_name]


def _rebuild_mp(mp, updates):
    """Return a new MetaparContainer with `updates` merged on top of `mp`.

    Greek codepoint guard: ϕ (U+03D5) and φ (U+03C6) are visually identical but
    map to different identifiers — keep only the one already in the base.
    """
    _base = mp._asdict()
    _full_dict = {**_base, **updates}
    if "ϕ" in _full_dict and "φ" in _full_dict:
        if "φ" in _base and "ϕ" not in _base:
            _full_dict["φ"] = _full_dict.get("φ", _full_dict.get("ϕ"))
            del _full_dict["ϕ"]
        else:
            _full_dict["ϕ"] = _full_dict.get("ϕ", _full_dict.get("φ"))
            del _full_dict["φ"]
    _MetaparContainer = namedtuple("MetaparContainer", list(_full_dict.keys()))
    return _MetaparContainer(**_full_dict)


def _activate_stage_mp(mp, stage, fit2_constant_overrides=None):
    """Switch `mp` to the correct gene subset and prior tensors for `stage`."""
    if stage == "fit_1":
        updates = dict(
            S=mp.S_fit1,
            U=mp.U_fit1,
            Ng=mp.Ng_fit1,
            μγ=mp.μγ_fit1,
            σγ=mp.σγ_fit1,
            μβ=mp.μβ_fit1,
            σβ=mp.σβ_fit1,
            fit1_gene_names=mp.fit1_gene_names,
        )
    elif stage == "fit_2":
        updates = dict(
            S=mp.S_fit2,
            U=mp.U_fit2,
            P=mp.P_fit2,
            Ng=mp.Ng_fit2,
            Ng_p=mp.Ng_p_fit2,
            μγ=mp.μγ_fit2,
            σγ=mp.σγ_fit2,
            μβ=mp.μβ_fit2,
            σβ=mp.σβ_fit2,
            μA=mp.μA_fit2,
            σA=mp.σA_fit2,
            μB=mp.μB_fit2,
            σB=mp.σB_fit2,
            fit2_gene_names=mp.fit2_gene_names,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    if stage == "fit_2" and fit2_constant_overrides:
        updates.update({k: v for k, v in fit2_constant_overrides.items() if v is not None})

    return _rebuild_mp(mp, updates)


def _reduce_samples(x, mode="mean"):
    """Aggregate posterior samples along the leading sample dimension."""
    if mode == "mean":
        return x.mean(0)
    if mode == "median":
        return x.median(0).values
    if mode == "sample":
        return x[0]
    raise ValueError(f"Unknown reduction mode: {mode}")


def _validate_fit1_predictive_shapes(pps_fit1, mp_obj):
    if pps_fit1 is None:
        raise ValueError("fit1 predictive output is None")
    for key in ["ElogS", "ElogU"]:
        if key in pps_fit1 and torch.is_tensor(pps_fit1[key]):
            if pps_fit1[key].shape[-2] != mp_obj.Ng or pps_fit1[key].shape[-1] != mp_obj.Nc:
                raise ValueError(f"fit1 {key} shape mismatch: expected (..., Ng, Nc)=(*,{mp_obj.Ng},{mp_obj.Nc}), got {tuple(pps_fit1[key].shape)}")


def _validate_fit2_override_shapes(overrides, mp_obj):
    expected_dtype = mp_obj.S_fit2.dtype if hasattr(mp_obj, "S_fit2") else mp_obj.S.dtype
    expected_device = mp_obj.device

    for key in ["ElogS_const_fit2", "ElogU_const_fit2"]:
        val = overrides.get(key)
        if torch.is_tensor(val):
            if val.shape[-2] != mp_obj.Ng_fit2 or val.shape[-1] != mp_obj.Nc:
                raise ValueError(f"{key} shape mismatch: expected (Ng_fit2, Nc)=({mp_obj.Ng_fit2},{mp_obj.Nc}), got {tuple(val.shape)}")
            if val.dtype != expected_dtype:
                raise TypeError(f"{key} dtype mismatch: expected {expected_dtype}, got {val.dtype}")
            if val.device != expected_device:
                raise TypeError(f"{key} device mismatch: expected {expected_device}, got {val.device}")

    for key in ["nu_const_fit2", "logβg_const_fit2", "logγg_const_fit2", "shape_inv_const_fit2"]:
        val = overrides.get(key)
        if torch.is_tensor(val):
            if val.shape[-2] != mp_obj.Ng_fit2:
                raise ValueError(f"{key} shape mismatch: expected first non-sample dim Ng_fit2={mp_obj.Ng_fit2}, got {tuple(val.shape)}")


def _validate_fit2_predictive_shapes(pps_fit2, mp_obj):
    if pps_fit2 is None:
        raise ValueError("fit2 predictive output is None")
    for key in ["ElogS", "ElogU", "ElogP"]:
        if key in pps_fit2 and torch.is_tensor(pps_fit2[key]):
            if pps_fit2[key].shape[-2] != mp_obj.Ng or pps_fit2[key].shape[-1] != mp_obj.Nc:
                raise ValueError(f"fit2 {key} shape mismatch: expected (..., Ng, Nc)=(*,{mp_obj.Ng},{mp_obj.Nc}), got {tuple(pps_fit2[key].shape)}")


def _validate_predictive_full_observation_mode(mp_obj, stage):
    """Ensure batching is disabled before running posterior predictive (requires full dataset)."""
    if getattr(mp_obj, "use_batching", False):
        raise ValueError(f"{stage} predictive mp must disable batching (use_batching=False)")
    c_attr = f"{stage}_cell_batch_size"
    g_attr = f"{stage}_gene_batch_size"
    c_bs = getattr(mp_obj, c_attr, None)
    g_bs = getattr(mp_obj, g_attr, None)
    if c_bs not in (None, -1):
        raise ValueError(f"{stage} predictive cell batch size must be None/-1, got {c_bs}")
    if g_bs not in (None, -1):
        raise ValueError(f"{stage} predictive gene batch size must be None/-1, got {g_bs}")


def _run_backend(model_full, mp_active, backend, svi_cfg, mcmc_cfg, stage=None):
    """Dispatch inference to SVI or MCMC."""
    from pyro.infer import Predictive
    from pyro.infer.autoguide import AutoDiagonalNormal, init_to_median
    from pyro.infer.mcmc import NUTS, MCMC

    from veloline.utils import fit_SVI

    backend = (backend or "svi").lower()

    if backend == "svi":
        res = fit_SVI(
            model_full,
            AutoDiagonalNormal,
            init_to_median,
            mp_active,
            initialization=True,
            n_inits=svi_cfg["n_inits"],
            warmup=svi_cfg["warmup"],
            n_iter=svi_cfg["n_iter"],
            lr=svi_cfg["lr"],
            betas=svi_cfg["betas"],
            stage=stage,
        )
        return {"backend": "svi", "res": res}

    if backend == "mcmc":
        pyro.clear_param_store()
        nuts = NUTS(
            model_full,
            target_accept_prob=float(mcmc_cfg.get("target_accept_prob", 0.8)),
            max_tree_depth=int(mcmc_cfg.get("max_tree_depth", 8)),
        )
        mcmc = MCMC(
            nuts,
            num_samples=int(mcmc_cfg.get("num_samples", 200)),
            warmup_steps=int(mcmc_cfg.get("warmup_steps", 200)),
        )
        mcmc.run(mp_active)
        return {"backend": "mcmc", "mcmc": mcmc}

    raise ValueError(f"Unknown inference backend: {backend}")
