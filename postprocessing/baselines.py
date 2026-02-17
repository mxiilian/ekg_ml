"""
Baseline-Imputations fuer EKG-NaN-Reparatur.

Alle Funktionen erwarten:
- x_filled: (B, C, T) Float Tensor, NaNs bereits als 0 gefuellt
- mask: (B, C, T) Float Tensor, 1=vorhanden, 0=fehlt

Rueckgabe:
- y_hat: (B, C, T) Float Tensor im gleichen Normalisierungsraum wie x_filled
"""

from typing import Optional
import numpy as np
import torch


def zero_fill(x_filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Baseline: keine Aenderung, fehlende Werte bleiben 0."""
    return x_filled


def mean_fill(x_filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Baseline: pro Lead den Mittelwert der vorhandenen Werte einsetzen."""
    mask_f = mask.float()
    sums = (x_filled * mask_f).sum(dim=-1, keepdim=True)
    counts = mask_f.sum(dim=-1, keepdim=True)
    means = torch.where(counts > 0, sums / counts, torch.zeros_like(sums))
    return torch.where(mask_f > 0, x_filled, means)


def _forward_fill_1d(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = values.copy()
    last = None
    for i in range(values.shape[0]):
        if valid[i]:
            last = values[i]
        elif last is not None:
            out[i] = last
    # If the series starts with missing values, keep them for now and
    # backfill with the first valid value if available.
    if valid.any():
        first_idx = int(np.argmax(valid))
        out[:first_idx] = values[first_idx]
    else:
        out[:] = 0.0
    return out


def locf_fill(x_filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Baseline: Last Observation Carried Forward (LOCF)."""
    x_np = x_filled.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.5
    bsz, channels, length = x_np.shape
    out = np.zeros_like(x_np)

    for b in range(bsz):
        for c in range(channels):
            out[b, c] = _forward_fill_1d(x_np[b, c], mask_np[b, c])

    return torch.from_numpy(out).to(device=x_filled.device, dtype=x_filled.dtype)


def _linear_interp_1d(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return np.zeros_like(values)
    idx = np.arange(values.shape[0])
    if valid.sum() == 1:
        fill_val = values[valid][0]
        return np.full_like(values, fill_val)
    out = values.copy()
    out[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return out


def linear_interp_fill(x_filled: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Baseline: lineare Interpolation pro Lead."""
    x_np = x_filled.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy() > 0.5
    bsz, channels, length = x_np.shape
    out = np.zeros_like(x_np)

    for b in range(bsz):
        for c in range(channels):
            out[b, c] = _linear_interp_1d(x_np[b, c], mask_np[b, c])

    return torch.from_numpy(out).to(device=x_filled.device, dtype=x_filled.dtype)


def apply_baseline(
    baseline: str,
    x_filled: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Dispatch fuer Baselines."""
    key = baseline.lower().strip()
    if key == "zero":
        return zero_fill(x_filled, mask)
    if key == "mean":
        return mean_fill(x_filled, mask)
    if key == "locf":
        return locf_fill(x_filled, mask)
    if key == "linear":
        return linear_interp_fill(x_filled, mask)
    raise ValueError(f"Unknown baseline: {baseline}")