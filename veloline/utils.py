"""Section 3 — utility & fitting functions, plus batching helpers from §7.

Contents (notebook cell 10 + the helper portion of cell 32):
  circ_median, check_model, squeeze_left
  init_try, find_best_seed
  FitResults, live_plot, fit_SVI
  _set/get_active_batch_subsamples, _normalize_batch_size,
  _deterministic_axis_chunks, _deterministic_epoch_subsamples
  _get_batch_sizes, _index_first, _index_last, _select_obs, _decaying_baseline
"""

import copy
import time
import collections

import numpy as np
import torch
import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import init_to_median

import pycircstat
import matplotlib.pyplot as plt
from IPython.display import clear_output

from veloline.metaparams import device


# ── UTILITIES ─────────────────────────────────────────────────────────────────

def circ_median(val, axis=0):
    """Circular median for angular pseudotime ϕ (handles wrap-around at 2π)."""
    return pycircstat.percentile(val, 50, np.array([0.0]), axis=axis)


def check_model(model, *args):
    """Dry-run a Pyro model, print plate shapes and render the graph."""
    pyro.clear_param_store()
    trace = poutine.trace(model).get_trace(*args)
    print(trace.format_shapes())
    return pyro.render_model(model, model_args=args)


def squeeze_left(a):
    """Remove all leading size-1 dimensions from `a` (e.g. (1,1,Ng,Nc) → (Ng,Nc))."""
    for i, s in enumerate(a.shape):
        if s == 1:
            pass
        else:
            final_shape = a.shape[i:]
            break
    return a.reshape(final_shape)


def init_try(seed, model, guide, optim, elbo, warmup, *args):
    """Run `warmup` SVI steps with a fixed seed and return the loss; used to compare initialisations."""
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    svi = SVI(model, guide, optim, loss=elbo)
    for _ in range(warmup):
        svi.step(*args)
    loss = svi.loss(model, guide, *args)
    pyro.clear_param_store()
    return loss


def find_best_seed(model, guide_type, optim, elbo, init_loc_fn, *args, n_inits=20, warmup=5):
    """Grid-search over `n_inits` random seeds; returns (best_seed, all_losses)."""
    pyro.clear_param_store()
    losses = []
    for seed in range(n_inits):
        guide = guide_type(model, init_loc_fn=init_loc_fn)
        loss = init_try(seed, model, guide, optim, elbo, warmup, *args)
        pyro.clear_param_store()
        losses.append(loss)
        print(f"Initialization Attempt with Seed={seed} - Loss was {loss}")
    pyro.clear_param_store()
    return np.argmin(losses), losses


class FitResults(object):
    pass


def live_plot(data_dict, figsize=(16, 5), title=''):
    """In-notebook updating ELBO curve — full history (log) + last-200 zoom."""
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.subplot(121)
    for label, data in data_dict.items():
        if isinstance(label, int):
            plt.scatter(np.full(len(data), label), data, marker="X", s=50, c="r")
        else:
            plt.plot(data, label=label, c="C0")
    plt.axhline(np.min(data), c="r")
    plt.title("All iterations log scale ")
    plt.grid(True)
    plt.xlabel('epoch')
    plt.yscale("log")
    plt.legend(loc='upper right')

    plt.subplot(122)
    for label, data in data_dict.items():
        if isinstance(label, int):
            continue
        plt.plot(data, label=label, c="C2")
    plt.title("Last 100 iterations")
    plt.axhline(np.min(data), c="r")
    plt.xlim(len(data) - 200, len(data))
    plt.ylim(np.min(data) - 10, np.max(data[-200:]))
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.show()


# ── Batching state (set per-step by the SVI loop, read by Pyro models) ───────

_ACTIVE_BATCH_SUBSAMPLES = {}


def _clear_active_batch_subsamples():
    global _ACTIVE_BATCH_SUBSAMPLES
    _ACTIVE_BATCH_SUBSAMPLES = {}


def _set_active_batch_subsamples(stage, cell_idx=None, gene_idx=None):
    """Register the active cell/gene index tensors for the current SVI step."""
    global _ACTIVE_BATCH_SUBSAMPLES
    if stage is None:
        _ACTIVE_BATCH_SUBSAMPLES = {}
        return
    _ACTIVE_BATCH_SUBSAMPLES = {(stage, "cells"): cell_idx, (stage, "genes"): gene_idx}


def _get_active_batch_subsample(stage, axis):
    return _ACTIVE_BATCH_SUBSAMPLES.get((stage, axis))


def _normalize_batch_size(bs, n_total):
    """Coerce a batch-size value to a valid int, or None if it covers the full dataset."""
    if bs is None:
        return None
    try:
        bs = int(bs)
    except Exception:
        return None
    if bs <= 0 or bs >= int(n_total):
        return None
    return bs


def _deterministic_axis_chunks(n_total, batch_size, n_batches, epoch_seed, dev):
    """Partition `n_total` items into `n_batches` deterministic random chunks using `epoch_seed`."""
    if batch_size is None:
        return [None] * n_batches
    rng = np.random.default_rng(int(epoch_seed))
    perm = rng.permutation(int(n_total))
    chunks = np.array_split(perm, n_batches)
    out = []
    for ch in chunks:
        if len(ch) == 0:
            out.append(torch.empty((0,), dtype=torch.long, device=dev))
        else:
            out.append(torch.as_tensor(ch, dtype=torch.long, device=dev))
    return out


def _deterministic_epoch_subsamples(mp_obj, stage, epoch_idx, base_seed=0):
    """Pre-compute cell and gene index batches for one epoch.

    Returns (n_batches, cell_chunks, gene_chunks) where each chunk is a list of
    LongTensors (or None for full-batch axes).
    """
    if not getattr(mp_obj, "use_batching", False):
        return 1, [None], [None]

    if stage == "fit_1":
        cell_bs = _normalize_batch_size(getattr(mp_obj, "fit_1_cell_batch_size", None), mp_obj.Nc)
        gene_bs = _normalize_batch_size(getattr(mp_obj, "fit_1_gene_batch_size", None), mp_obj.Ng)
        stage_bias = 0
    elif stage == "fit_2":
        cell_bs = _normalize_batch_size(getattr(mp_obj, "fit_2_cell_batch_size", None), mp_obj.Nc)
        gene_bs = _normalize_batch_size(getattr(mp_obj, "fit_2_gene_batch_size", None), mp_obj.Ng)
        stage_bias = 10_000
    else:
        return 1, [None], [None]

    if cell_bs is None and gene_bs is None:
        return 1, [None], [None]

    n_cell_batches = 1 if cell_bs is None else int(np.ceil(mp_obj.Nc / cell_bs))
    n_gene_batches = 1 if gene_bs is None else int(np.ceil(mp_obj.Ng / gene_bs))
    n_batches = max(n_cell_batches, n_gene_batches)

    dev = mp_obj.device if hasattr(mp_obj, "device") else torch.device("cpu")
    cell_chunks = _deterministic_axis_chunks(mp_obj.Nc, cell_bs, n_batches, base_seed + stage_bias + epoch_idx, dev)
    gene_chunks = _deterministic_axis_chunks(mp_obj.Ng, gene_bs, n_batches, base_seed + stage_bias + 1_000_000 + epoch_idx, dev)
    return n_batches, cell_chunks, gene_chunks


def fit_SVI(model, guide_type, init_loc_fn, *args, initialization=False, n_inits=20, warmup=5,
            n_iter=3000, lr=0.01, betas=(0.8, 0.99), optim=None, stage=None,
            timing_cv_warn_threshold=0.20):
    """Main SVI training loop with optional seed search, batching, and best-state tracking."""
    pyro.clear_param_store()
    ploting_data = collections.defaultdict(list)

    def init_to_median_device(site):
        value = init_to_median(site)
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return value
    init_loc_fn = init_to_median_device

    print("Clear Parameter Store")
    if optim is None:
        optim = pyro.optim.Adam({'lr': lr, 'betas': betas})
    elbo = Trace_ELBO()
    base_seed = 0

    if initialization:
        best_seed, init_losses = find_best_seed(model, guide_type, optim, elbo, init_loc_fn, *args,
                                                n_inits=n_inits, warmup=warmup)
        ploting_data[warmup] = init_losses
        pyro.set_rng_seed(int(best_seed))
        base_seed = int(best_seed)

    guide = guide_type(model, init_loc_fn=init_loc_fn)
    svi = SVI(model, guide, optim, loss=elbo)
    print("Initializing SVI...")

    mp_obj = args[0] if len(args) > 0 else None
    n_batches, cell_chunks, gene_chunks = (
        _deterministic_epoch_subsamples(mp_obj, stage, epoch_idx=0, base_seed=base_seed)
        if mp_obj is not None else (1, [None], [None])
    )
    _set_active_batch_subsamples(stage, cell_chunks[0], gene_chunks[0])

    t0 = time.time()
    init_loss = svi.loss(model, guide, *args)
    loss = svi.step(*args)
    t1 = time.time()

    init_cell_n = mp_obj.Nc if (mp_obj is not None and cell_chunks[0] is None) else (int(cell_chunks[0].numel()) if mp_obj is not None else 1)
    init_gene_n = mp_obj.Ng if (mp_obj is not None and gene_chunks[0] is None) else (int(gene_chunks[0].numel()) if mp_obj is not None else 1)
    init_work_n = max(1, init_cell_n * init_gene_n)
    print(f"ELBO loss at initialization: {init_loss}  -  Single step takes {t1-t0:.2e} seconds ({(t1-t0)/init_work_n:.2e} sec/work-item)")

    losses = []
    best_loss = float('inf')
    pmstore_minstate = None
    step_times = []

    for i in range(1, n_iter):
        n_batches, cell_chunks, gene_chunks = (
            _deterministic_epoch_subsamples(mp_obj, stage, epoch_idx=i, base_seed=base_seed)
            if mp_obj is not None else (1, [None], [None])
        )

        step_loss = 0.0
        t_step_start = time.time()
        for b in range(n_batches):
            _set_active_batch_subsamples(stage, cell_chunks[b], gene_chunks[b])
            step_loss += svi.step(*args)
        step_times.append(time.time() - t_step_start)

        losses.append(step_loss)

        if step_loss < best_loss:
            best_loss = step_loss
            pmstore_minstate = copy.deepcopy(pyro.get_param_store().get_state())

        if i % 40 == 0:
            ploting_data["ELBO"] = losses[:]
            live_plot(ploting_data, title=f"Step {i}/{n_iter}")

    if pmstore_minstate is not None:
        pyro.get_param_store().set_state(pmstore_minstate)

    if len(step_times) > 1:
        cv = np.std(step_times) / (np.mean(step_times) + 1e-12)
        if cv > timing_cv_warn_threshold:
            print(f"Warning: step timing CV={cv:.2f} > threshold={timing_cv_warn_threshold}")

    result = FitResults()
    result.guide = guide
    result.losses = losses
    result.best_loss = best_loss
    result.step_times = step_times
    return result


# ── Batching helpers used by Pyro models (cell 32 head) ───────────────────────

def _get_batch_sizes(mp, stage):
    """Return (cell_batch_size, gene_batch_size) or (None, None) when batching is disabled."""
    if not getattr(mp, "use_batching", False):
        return None, None
    cell_bs = getattr(mp, f"{stage}_cell_batch_size", -1)
    gene_bs = getattr(mp, f"{stage}_gene_batch_size", -1)
    if cell_bs is not None and (cell_bs <= 0 or cell_bs >= mp.Nc):
        cell_bs = None
    if gene_bs is not None and (gene_bs <= 0 or gene_bs >= mp.Ng):
        gene_bs = None
    return cell_bs, gene_bs


def _index_first(x, idx):
    """Index `x` along its first axis (gene axis) using `idx`; no-op if idx is None."""
    return x if idx is None else torch.index_select(x, 0, idx)


def _index_last(x, idx):
    """Index `x` along its last axis (cell axis) using `idx`; no-op if idx is None."""
    return x if idx is None else torch.index_select(x, -1, idx)


def _select_obs(obs, gene_idx=None, cell_idx=None):
    """Slice the (Ng, Nc) observation tensor to the active gene/cell mini-batch."""
    out = obs
    if gene_idx is not None:
        out = torch.index_select(out, 0, gene_idx)
    if cell_idx is not None:
        out = torch.index_select(out, 1, cell_idx)
    return out


def _decaying_baseline(mp, ϕ, branch):
    """Optional decaying count-rate baseline along ϕ (exponential or linear). Disabled by default."""
    if not getattr(mp, "baseline_enabled", False):
        return 0.0
    baseline_branch = getattr(mp, "baseline_branch", "all")
    if baseline_branch not in ("all", branch):
        return 0.0

    family = getattr(mp, "baseline_family", "exponential")
    amp = torch.as_tensor(float(getattr(mp, "baseline_amplitude", 0.0)), device=device)
    decay_rate = torch.as_tensor(float(getattr(mp, "baseline_decay_rate", 1.0)), device=device)
    floor = torch.as_tensor(float(getattr(mp, "baseline_floor", 0.0)), device=device)

    ϕn = ((ϕ.squeeze() - mp.ø) / (mp.χ - mp.ø + 1e-8)).to(device)
    if family == "linear":
        baseline = floor + amp * torch.clamp(1.0 - decay_rate * ϕn, min=0.0).to(device)
    else:
        baseline = floor + amp * torch.exp(-decay_rate * ϕn).to(device)
    return baseline.unsqueeze(0)
