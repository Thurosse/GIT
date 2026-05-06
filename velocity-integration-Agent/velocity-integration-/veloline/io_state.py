"""Run directory + state save/load helpers.

Each pipeline run lives in `results/run_<UTC>_<short-hash>/`. The first stage
(setup) mints the directory and writes baseline state; subsequent stages
(inference, analysis) load from there and append their own outputs.

`results/latest.txt` always points at the most recent run directory (Windows-friendly
alternative to a symlink).
"""

import os
import json
import time
import uuid
import shutil
from collections import namedtuple
from datetime import datetime, timezone

import numpy as np
import torch
import scanpy as sc

from veloline.mp_builder import MP_SCHEMA_VERSION


PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")


def _short_hash():
    return uuid.uuid4().hex[:6]


def mint_run_dir(name=None):
    """Create a new run directory under `results/` and update `latest.txt`."""
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    suffix = name or _short_hash()
    run_name = f"run_{ts}_{suffix}"
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    for sub in ("state", "plots", "metrics", "logs", "manifests"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    with open(os.path.join(RESULTS_ROOT, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(run_name)
    return run_dir


def latest_run_dir():
    """Return the most recent run directory (from `latest.txt`), or raise."""
    pointer = os.path.join(RESULTS_ROOT, "latest.txt")
    if not os.path.exists(pointer):
        raise FileNotFoundError(
            f"No latest.txt at {pointer}. Run the setup stage first to mint a run directory."
        )
    with open(pointer, encoding="utf-8") as f:
        run_name = f.read().strip()
    run_dir = os.path.join(RESULTS_ROOT, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"latest.txt points at missing directory: {run_dir}")
    return run_dir


def resolve_run_dir(run=None):
    """Resolve `run` to an absolute directory. `run='latest'` or None → `latest.txt`."""
    if run in (None, "latest"):
        return latest_run_dir()
    if os.path.isabs(run):
        return run
    candidate = os.path.join(RESULTS_ROOT, run)
    return candidate if os.path.isdir(candidate) else run


# ── State save/load ───────────────────────────────────────────────────────────

def save_mp(mp, run_dir):
    """Serialize `mp` as a dict-of-tensors plus MP_SCHEMA_VERSION, plus a JSON sidecar
    with the human-readable scalar fields.
    """
    state = {"_schema_version": MP_SCHEMA_VERSION, "_fields": list(mp._fields)}
    state.update(mp._asdict())
    torch.save(state, os.path.join(run_dir, "state", "mp.pt"))

    meta = {}
    for f in mp._fields:
        v = getattr(mp, f)
        if isinstance(v, (int, float, bool, str)) or v is None:
            meta[f] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
            meta[f] = list(v)
        elif isinstance(v, np.ndarray) and v.ndim <= 1 and v.size <= 50:
            meta[f] = v.tolist()
    with open(os.path.join(run_dir, "state", "mp_meta.json"), "w", encoding="utf-8") as out:
        json.dump(meta, out, indent=2, default=str)


def load_mp(run_dir):
    """Reconstruct `mp` from `state/mp.pt`. Refuses to load if schema version differs."""
    state = torch.load(os.path.join(run_dir, "state", "mp.pt"), map_location="cpu", weights_only=False)
    schema = state.pop("_schema_version", None)
    fields = state.pop("_fields", None)
    if schema != MP_SCHEMA_VERSION:
        raise RuntimeError(
            f"mp schema mismatch: stored v{schema} but veloline expects v{MP_SCHEMA_VERSION}. "
            "Re-run the setup stage to regenerate state/mp.pt."
        )
    if fields is None:
        fields = list(state.keys())
    MetaparContainer = namedtuple("MetaparContainer", fields)
    return MetaparContainer(**{k: state[k] for k in fields})


def save_adata(adata, path):
    """Write an AnnData to `path` using the native h5ad format."""
    adata.write_h5ad(path)


def load_adata(path):
    return sc.read_h5ad(path)


def save_workflow(workflow, run_dir):
    """JSON snapshot of the resolved MODEL_WORKFLOW used in this run."""
    with open(os.path.join(run_dir, "state", "model_workflow.json"), "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, default=str)


def save_rng_state(run_dir, seed, use_gpu):
    """Capture pyro/torch/numpy RNG state + device flag."""
    state = {
        "pyro_seed": int(seed),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "device": "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu",
        "use_gpu": bool(use_gpu),
    }
    torch.save(state, os.path.join(run_dir, "state", "rng_state.pt"))


def load_rng_state(run_dir):
    return torch.load(os.path.join(run_dir, "state", "rng_state.pt"), map_location="cpu", weights_only=False)


def restore_rng(state):
    """Re-issue device + RNG state from the dict returned by `load_rng_state`."""
    import pyro
    torch.set_default_device(state["device"])
    pyro.set_rng_seed(state["pyro_seed"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def save_posteriors(posteriors, path):
    """Write a posterior dict (containing torch tensors) via torch.save."""
    torch.save(posteriors, path)


def load_posteriors(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def freeze_metaparams(run_dir):
    """Snapshot `veloline/metaparams.py` into `manifests/metaparams_snapshot.py`."""
    src = os.path.join(PROJECT_ROOT, "veloline", "metaparams.py")
    dst = os.path.join(run_dir, "manifests", "metaparams_snapshot.py")
    shutil.copyfile(src, dst)


def write_manifest(run_dir, extra=None):
    """Write `manifests/run_manifest.json` capturing pkg versions + GPU info + timing slot."""
    import torch as _torch
    try:
        import pyro as _pyro
        pyro_v = _pyro.__version__
    except Exception:
        pyro_v = None
    info = {
        "run_dir": os.path.basename(run_dir),
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "torch": _torch.__version__,
        "pyro": pyro_v,
        "cuda_available": _torch.cuda.is_available(),
        "cuda_device": (_torch.cuda.get_device_name(0) if _torch.cuda.is_available() else None),
        "mp_schema_version": MP_SCHEMA_VERSION,
    }
    if extra:
        info.update(extra)
    with open(os.path.join(run_dir, "manifests", "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)


def append_log(run_dir, stage, msg):
    """Append a timestamped line to `logs/<stage>.log`."""
    path = os.path.join(run_dir, "logs", f"{stage}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
