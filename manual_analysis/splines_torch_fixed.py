import numpy as np
import torch
from scipy.interpolate import splder, splantider
from typing import Union, Optional


def spline_prep(lower_bound=0, upper_bound=1, df=6, degree=3):
    """Returns the nodes and degree of of a spline"""
    order = degree + 1
    n_inner_knots = df - order
    inner_knots = np.linspace(lower_bound, upper_bound, n_inner_knots + 2)[1:-1]
    all_knots = np.concatenate(
        ([lower_bound] * order, inner_knots, [upper_bound] * order)
    )
    t = all_knots
    k = degree
    return t, k


def torch_B(x, k, i, t):
    device = x.device
    if k == 0:
        res = torch.ones(len(x), len(i), dtype=x.dtype, device=device)
        res[~((t[i] <= x) & (x <= t[i + 1]))] = 0.0
        return res

    c1 = torch.zeros(len(x), len(i), dtype=x.dtype, device=device)
    c2 = torch.zeros(len(x), len(i), dtype=x.dtype, device=device)
    
    bool_ = ~(t[i + k] == t[i])
    c1[:, ~bool_] = 0
    c1[:, bool_] = (
        (x - t[i][bool_]) / (t[i + k] - t[i])[bool_] * torch_B(x, k - 1, i, t)[:, bool_]
    )

    bool_ = ~(t[i + k + 1] == t[i + 1])
    c2[:, ~bool_] = 0
    c2[:, bool_] = (
        (t[i + k + 1][bool_] - x)
        / (t[i + k + 1] - t[i + 1])[bool_]
        * torch_B(x, k - 1, i + 1, t)[:, bool_]
    )

    return c1 + c2


def tvect_B(x, k, i, t):
    """Fully vectorial, but less efficient version"""
    device = x.device
    if k == 0:
        return torch.where(
            (t[i] <= x) & (x <= t[i + 1]),
            torch.tensor([1.0], dtype=x.dtype, device=device),
            torch.tensor([0.0], dtype=x.dtype, device=device),
        )
    c1 = torch.where(
        t[i + k] == t[i],
        torch.tensor([0.0], dtype=x.dtype, device=device),
        (x - t[i]) / (t[i + k] - t[i] + 1e-16) * tvect_B(x, k - 1, i, t),
    )

    c2 = torch.where(
        t[i + k + 1] == t[i + 1],
        torch.tensor([0.0], dtype=x.dtype, device=device),
        (t[i + k + 1] - x)
        / (t[i + k + 1] - t[i + 1] + 1e-16)
        * tvect_B(x, k - 1, i + 1, t),
    )
    return c1 + c2


def derivative(t, k, c=None, nu=1):
    if c is None:
        n = len(t) - k - 1
        c = np.eye(n, dtype=t.dtype)
    ct = len(t) - len(c)
    if ct > 0:
        c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
    tck = splder((t, c, k), nu)
    return tck


def antiderivative(t, k, c=None, nu=1):
    if c is None:
        n = len(t) - k - 1
        c = np.eye(n, dtype=t.dtype)
    ct = len(t) - len(c)
    if ct > 0:
        c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
    tck = splantider((t, c, k), nu)
    return tck


def torch_spline_basis(
    x: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prepend: Optional[int] = None,
) -> torch.Tensor:
    
    # Ensure x is a tensor to establish device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    device = x.device

    if isinstance(t, np.ndarray):
        t = torch.tensor(t, device=device)
    else:
        t = t.to(device)

    if c is not None:
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, device=device)
        else:
            c = c.to(device)

    n = len(t) - k - 1
    # arange needs to be on device
    indices = torch.arange(n, device=device)
    D = torch_B(x[:, None], k, indices, t.type(x.dtype))
    
    if c is not None:
        D = D @ c.type(D.dtype)[:n, :]

    if prepend is not None:
        if prepend == 0:
            fill_val = 0.0
        elif prepend == 1:
            fill_val = 1.0
        else:
            fill_val = float(prepend)
        
        prefix = torch.full((D.shape[0], 1), fill_val, dtype=D.dtype, device=device)
        return torch.column_stack([prefix, D])
    
    return D


def torch_spline_basis_primitive(
    x: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    lower_bound: Optional[float] = 0.0,
    prepend: Optional[int] = None,
) -> torch.Tensor:
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    device = x.device

    D = torch_spline_basis(x, t, k, c, prepend=None)

    if lower_bound is not None:
        lb = torch.full((1,), float(lower_bound), dtype=x.dtype, device=device)
        D0 = torch_spline_basis(lb, t, k, c, prepend=None)
        D = D - D0

    if prepend is not None:
        if prepend == 0:
            fill_val = 0.0
        elif prepend == 1:
            fill_val = 1.0
        else:
            fill_val = float(prepend)
        
        prefix = torch.full((D.shape[0], 1), fill_val, dtype=D.dtype, device=device)
        return torch.column_stack([prefix, D])
    
    return D


def tvect_spline_basis(x, t, k, c=None, prepend=None):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    device = x.device
    
    if isinstance(t, np.ndarray):
        t = torch.tensor(t, device=device)
    else:
        t = t.to(device)

    n = len(t) - k - 1
    indices = torch.arange(n, device=device)
    D = tvect_B(x[:, None], k, indices, t.type(x.dtype))
    
    if c is not None:
        if isinstance(c, np.ndarray):
            c = torch.tensor(c, device=device)
        else:
            c = c.to(device)
        D = D @ c.type(D.dtype)[:n, :]

    if prepend is not None:
        fill_val = float(prepend)
        prefix = torch.full((D.shape[0], 1), fill_val, dtype=D.dtype, device=device)
        return torch.column_stack([prefix, D])
    
    return D


def torch_spline_basis_2d(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prepend: Optional[int] = None,
) -> torch.Tensor:
    
    Dx = torch_spline_basis(x, t, k)
    Dy = torch_spline_basis(y, t, k)
    device = Dx.device
    
    Dxy = Dy.repeat((1, Dx.shape[1])) * Dx.repeat_interleave(Dy.shape[1], dim=1)
    
    if prepend is not None:
        fill_val = float(prepend)
        prefix = torch.full((Dxy.shape[0], 1), fill_val, dtype=Dxy.dtype, device=device)
        return torch.column_stack([prefix, Dxy])
    
    return Dxy


def torch_spline_basis_2d_der(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    t: Union[torch.Tensor, np.ndarray],
    tder: Union[torch.Tensor, np.ndarray],
    k: int = 3,
    kder: int = 2,
    c: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prepend: Optional[int] = None,
):
    Dx = torch_spline_basis(x, t, k)
    Dy = torch_spline_basis(y, t, k)
    Dxdx = torch_spline_basis(x, tder, kder, c)
    Dydy = torch_spline_basis(y, tder, kder, c)
    
    device = Dx.device

    Dxydy = Dydy.repeat((1, Dx.shape[1])) * Dx.repeat_interleave(Dydy.shape[1], dim=1)
    Dxydx = Dy.repeat((1, Dxdx.shape[1])) * Dxdx.repeat_interleave(Dy.shape[1], dim=1)
    
    if prepend is not None:
        fill_val = float(prepend)
        prefix_dy = torch.full((Dxydy.shape[0], 1), fill_val, dtype=Dxydy.dtype, device=device)
        prefix_dx = torch.full((Dxydx.shape[0], 1), fill_val, dtype=Dxydx.dtype, device=device)
        return (
            torch.column_stack([prefix_dy, Dxydy]),
            torch.column_stack([prefix_dx, Dxydx]),
        )
    
    return Dxydy, Dxydx