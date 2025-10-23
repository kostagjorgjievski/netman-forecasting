"""
src/train_eval.py

Training/validation utilities for time-series forecasting:

- Thin tqdm progress bar that auto-disables in non-TTY environments.
- Forward helpers to evaluate with BN/Dropout in eval mode while preserving model.train() state.
- Shape-robust target selection for models that emit [B,H,C] or [B,C,H].
- Training loop that computes *true epoch MSE* via sum-and-count (robust to NaNs/Infs).
- Evaluation that can also compute inverse-scaled metrics when (mu_t, sd_t) are provided.
- Naive last-value baseline for comparison.

Notes:
- All prints/logging preserved as in the original (including first-batch debug).
- No functional changes: arguments, defaults, control flow, and math are identical.
"""

from tqdm import tqdm
import torch, torch.nn as nn
import os, sys

# Whether this is "main" rank with an interactive stderr (used for progress display)
is_main = (os.environ.get("RANK", "0") == "0") and sys.stderr.isatty()


# ----------------------------
# Progress bar helper
# ----------------------------
def _make_pbar(it, desc):
    """
    Make a throttled tqdm progress bar that:
      - disables itself if stderr isn't a TTY, or if PROGRESS env var is off,
      - limits refresh rate to reduce overhead in fast iters.

    Env:
      PROGRESS=0|off|false|no  -> force-disable the bar.
    """
    disable = (not sys.stderr.isatty()) or os.environ.get("PROGRESS","auto").lower() in ("0","off","false","no")
    return tqdm(
        it, desc=desc, ncols=80, leave=False,
        mininterval=2.0,  # throttle refresh
        disable=disable
    )


# ----------------------------
# Forward/eval helpers
# ----------------------------
def _forward_evalmode(model, x, xmark, device):
    """
    Forward pass with model.eval() semantics (BN uses running stats, Dropout off),
    then restore the original training flag.

    Args:
      model:  torch.nn.Module
      x:      [B,L,D]  input window (scaled)
      xmark:  [B,L,*]  time markers (can be empty)
      device: torch.device

    Returns:
      out:    model output tensor (if tuple/list, returns first element)
    """
    was_training = model.training
    model.eval()  # BN uses running stats; Dropout disabled
    out = model(x, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
    model.train(was_training)  # restore original flag
    return out[0] if isinstance(out, (list, tuple)) else out


def _select_target(out, y):
    """
    Extract the 1-channel target slice from model output, handling multiple layouts:

      Expected:
        y:   [B, H, 1] (target)
        out: [B, H, C]  -> take last feature: out[..., -1:] => [B,H,1]
      Alternate:
        out: [B, C, H]  -> take last feature then transpose to [B,H,1]
      Fallback:
        out[..., -1:]   -> last channel along the last dim

    Args:
      out: model output
      y:   ground truth (for checking H dimension)

    Returns:
      out_ot: [B, H, 1] aligned with y
    """
    # y is [B, H, 1]; handle [B,H,C] and [B,C,H]
    if out.dim() == 3 and out.shape[1] == y.shape[1]:   # [B,H,C]
        return out[..., -1:]                            # [B,H,1]
    if out.dim() == 3 and out.shape[2] == y.shape[1]:   # [B,C,H]
        return out[:, -1:, :].transpose(1, 2)           # [B,H,1]
    return out[..., -1:]


def _unpack(batch):
    """Tuple-unpack a dataloader batch to (x, xmark, y)."""
    x, xmark, y = batch[0], batch[1], batch[2]
    return x, xmark, y


# ----------------------------
# Training (one epoch)
# ----------------------------
def train_epoch(model, loader, optim, device, clip=1.0):
    """
    One training epoch with robust MSE accounting:

      - Computes exact epoch MSE as (sum of squared errors) / (count of finite elements).
      - Explicitly masks NaN/Inf in both predictions and targets.
      - Optional gradient clipping by global norm.
      - Lightweight tqdm progress bar that auto-disables in non-interactive runs.

    Args:
      model:  torch.nn.Module in training mode
      loader: DataLoader yielding (x, xmark, y)
      optim:  torch.optim.Optimizer
      device: torch.device
      clip:   float or None; max grad-norm (global) if not None

    Returns:
      float: true epoch MSE over all finite elements.
    """
    model.train()
    se_total = 0.0  # sum of squared errors over epoch
    n_total  = 0    # number of finite elements accumulated
    printed_debug = False  # (placeholder for optional first-batch prints)

    pbar = _make_pbar(loader, "Training")
    for i, batch in enumerate(pbar):
        # Move batch to device
        x, xmark, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # Forward (train-mode)
        optim.zero_grad(set_to_none=True)
        out = model(x, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
        out = out[0] if isinstance(out, (list, tuple)) else out
        # out = _forward_evalmode(model, x, xmark, device)  # (kept commented as in original)

        # Robust target selection: handle [B,H,C] or [B,C,H] or fallback
        if out.dim() == 3 and out.shape[1] == y.shape[1]:
            out_ot = out[..., -1:]               # [B,H,1]
        elif out.dim() == 3 and out.shape[2] == y.shape[1]:
            out_ot = out[:, -1:, :].transpose(1, 2)  # [B,H,1]
        else:
            out_ot = out[..., -1:]

        # Mask non-finite values on both sides to protect reductions
        mask = torch.isfinite(out_ot) & torch.isfinite(y)
        n_el = y[mask].numel()
        if n_el == 0:
            # Nothing usable in this batch; skip without penalty
            continue

        # Exact batch MSE via sum & count (not mean-of-means)
        diff = out_ot[mask] - y[mask]
        batch_se = (diff * diff).sum()          # SUM of squared errors
        loss = batch_se / n_el                  # exact batch MSE scalar

        # Skip non-finite losses (defensive)
        if not torch.isfinite(loss):
            continue

        # Backprop + (optional) grad clip + step
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optim.step()

        # Accumulate epoch statistics
        se_total += float(batch_se.item())
        n_total  += n_el

        # Live epoch MSE in the pbar tail
        pbar.set_postfix_str(f"epoch_mse={se_total/max(n_total,1):.5f}")

    # True epoch MSE (handle empty epoch gracefully)
    return se_total / max(n_total, 1)


# ----------------------------
# Evaluation (validation/test)
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, mu_sd=None, show_pbar=False):
    """
    Evaluate MSE/MAE in scaled space; optionally also compute inverse-scaled metrics.

    Args:
      model:    torch.nn.Module
      loader:   DataLoader yielding (x, xmark, y)
      device:   torch.device
      mu_sd:    optional tuple (mu_t, sd_t) (tensors) for inverse-scaling the target
      show_pbar:bool; if True, show tqdm during evaluation

    Returns:
      (mse, mae)                      # scaled space
      or
      (mse, mae, mse_inv, mae_inv)    # if mu_sd provided (original scale)
    """
    model.eval()
    it = tqdm(loader, desc="Validate", ncols=80, leave=False) if show_pbar else loader

    mse_sum = 0.0; mae_sum = 0.0; n_sum = 0
    mse_inv_sum = 0.0; mae_inv_sum = 0.0

    use_inv = mu_sd is not None
    if use_inv:
        mu_t, sd_t = mu_sd
        j = -1  # last column is the target

    for batch in it:
        # Unpack + move to device
        x, xmark, y = _unpack(batch)
        x, xmark, y = x.to(device), xmark.to(device), y.to(device)

        # Forward (eval-mode already set by decorator)
        out = model(x, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
        out = out[0] if isinstance(out, (list, tuple)) else out
        # out = _forward_evalmode(model, x, xmark, device)  # (kept commented)
        out_ot = _select_target(out, y)  # [B,H,1]

        # Finite mask for robust reduction
        mask = torch.isfinite(out_ot) & torch.isfinite(y)
        n_el = y[mask].numel()
        if n_el == 0:
            continue

        # Scaled-space errors
        d = out_ot[mask] - y[mask]
        mse_sum += float((d * d).sum().item())
        mae_sum += float(d.abs().sum().item())
        n_sum   += n_el

        # Inverse-scaled metrics (if requested)
        if use_inv:
            out_inv = out_ot * sd_t[j] + mu_t[j]
            y_inv   = y      * sd_t[j] + mu_t[j]
            di = (out_inv[mask] - y_inv[mask])
            mse_inv_sum += float((di * di).sum().item())
            mae_inv_sum += float(di.abs().sum().item())

    mse = mse_sum / max(n_sum, 1)
    mae = mae_sum / max(n_sum, 1)

    if use_inv:
        return mse, mae, mse_inv_sum / max(n_sum, 1), mae_inv_sum / max(n_sum, 1)
    return mse, mae


# ----------------------------
# Naive baseline (last value)
# ----------------------------
@torch.no_grad()
def naive_last_value(loader, device, mu_sd=None):
    """
    Last-value baseline on scaled data (optionally inverse-scaled too).

    Prediction rule:
      yhat[b, t, 0] = x[b, -1, -1]   for all t in horizon

    Args:
      loader: DataLoader yielding (x, xmark, y)
      device: torch.device
      mu_sd:  optional (mu_t, sd_t) for inverse-scaling

    Returns:
      (mse, mae) or (mse, mae, mse_inv, mae_inv) if mu_sd provided.
    """
    mse_sum = 0.0; mae_sum = 0.0; n_sum = 0
    mse_inv_sum = 0.0; mae_inv_sum = 0.0
    use_inv = mu_sd is not None
    if use_inv:
        mu_t, sd_t = mu_sd
        j = -1  # last column index

    for batch in loader:
        x, xmark, y = _unpack(batch)
        x, y = x.to(device), y.to(device)

        # Predict the future by repeating the last observed target value
        last = x[:, -1:, -1:]       # [B,1,1] last target observation
        yhat = last.expand_as(y)    # [B,H,1] repeat across horizon

        # Robust masking
        mask = torch.isfinite(yhat) & torch.isfinite(y)
        n_el = y[mask].numel()
        if n_el == 0:
            continue

        # Scaled-space errors
        d = yhat[mask] - y[mask]
        mse_sum += float((d * d).sum().item())
        mae_sum += float(d.abs().sum().item())
        n_sum   += n_el

        # Inverse-scaled metrics (if requested)
        if use_inv:
            yhat_inv = yhat * sd_t[j] + mu_t[j]
            y_inv    = y    * sd_t[j] + mu_t[j]
            di = yhat_inv[mask] - y_inv[mask]
            mse_inv_sum += float((di * di).sum().item())
            mae_inv_sum += float(di.abs().sum().item())

    mse = mse_sum / max(n_sum, 1)
    mae = mae_sum / max(n_sum, 1)
    if use_inv:
        return mse, mae, mse_inv_sum / max(n_sum, 1), mae_inv_sum / max(n_sum, 1)
    return mse, mae
