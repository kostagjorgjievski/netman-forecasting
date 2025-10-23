"""
Sliding-window dataset and dataloader builder for time-series forecasting.

- Loads a CSV, keeps only numeric columns (last column is the target).
- Cleans data (finite-only with causal forward-fill; leading NaNs -> 0).
- Fits normalization (mu, sd) on the *first 70% of rows*, then applies to all rows.
- Constructs [L, D] → [H, 1] sliding windows, filtering any window with non-finite values.
- Builds train/val/test DataLoaders with time-ordered splits and leakage gaps.
- Returns loaders plus scaling stats (mu, sd, and target-channel mu_t, sd_t).
"""

import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple
import random

EPS = 1e-6  # small epsilon for std clamping


# ---------------------------
# Utilities
# ---------------------------
def _clean_matrix(arr: np.ndarray) -> np.ndarray:
    """
    Ensure a clean float32 matrix with finite values:
    - Convert to float32.
    - Mark non-finite as NaN.
    - Causally forward-fill NaNs column-wise.
    - Replace remaining leading NaNs/±inf with 0.

    Notes:
      * The causal forward fill uses a running maximum of "last valid row idx"
        per column, then gathers values from those positions.
      * Any prefix with no valid values stays NaN and is then zeroed.
    """
    arr = arr.astype(np.float32, copy=False)
    # Replace non-finite with NaN for forward-fill pass
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mask = np.isnan(arr)  # True where we need to fill

    # Build an index matrix of last-seen valid row per column:
    # If current row is valid, idx[row,col] = row, else 0.
    idx = np.where(~mask, np.arange(arr.shape[0])[:, None], 0)
    # Cumulative max down each column gives last valid index up to current row
    np.maximum.accumulate(idx, axis=0, out=idx)
    # Gather the last-seen valid values (undefined for leading NaNs → handled below)
    ff = arr[idx, np.arange(arr.shape[1])]
    filled = np.where(mask, ff, arr)

    # Turn any remaining NaN/±inf (e.g., leading segments) into zeros
    filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    return filled


def _fit_scale(train: np.ndarray):
    """
    Fit per-column standardization parameters on the provided training slice.

    Args:
      train: [Ttr, D] training rows only (first 70% by time in caller).

    Returns:
      mu: per-column mean
      sd: per-column std with tiny values clamped to 1.0 to avoid div-by-zero.
    """
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd < EPS] = 1.0  # guard against degenerate variance
    return mu, sd


def _apply_scale(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply standardization (x - mu) / sd elementwise (broadcasts over rows)."""
    return (x - mu) / sd

def _seed_worker(worker_id):
    """
    Worker seeding for deterministic-ish DataLoader behavior:
    derive a 32-bit seed from torch's worker seed and use it for NumPy/py.random.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------
# Dataset
# ---------------------------
class SlidingWindowDataset(Dataset):
    """
    Time-series windows from a normalized matrix X with shape [T, D].

    Assumptions:
      - Last column is the prediction target.
      - Only windows with all finite entries (inputs + targets) are kept.

    Returns per item:
      x:       [L, D]    (input window)
      x_mark:  [L, 0]    (placeholder; empty time markers)
      y:       [H, 1]    (future target horizon)
    """

    def __init__(self, data_scaled: np.ndarray, seq_len: int, pred_len: int):
        assert data_scaled.ndim == 2
        self.X = data_scaled.astype(np.float32, copy=False)
        self.L = seq_len
        self.H = pred_len

        # Precompute valid start indices for windows of length L+H
        T = len(self.X)
        self.idx = np.array([], dtype=np.int64)
        if T >= (self.L + self.H):
            k = self.L + self.H

            # Mark rows that are entirely finite (across all D features)
            finite_rows = np.isfinite(self.X).all(axis=1).astype(np.int32)

            # Sliding window sum of 'finite' flags over window size k using prefix sums
            csum = np.cumsum(finite_rows)
            left = np.concatenate(([0], csum[:-k]))  # prefix sum up to start-1
            right = csum[k - 1:]                     # prefix sum up to end
            win_sum = right - left                   # count of finite rows in each window

            # Valid windows have k finite rows (no NaNs/infs anywhere in [s:s+k))
            valid = (win_sum == k)
            idx_all = np.arange(0, T - k + 1, dtype=np.int64)
            self.idx = idx_all[valid]

    def __len__(self) -> int:
        # Number of valid window starting positions
        return int(self.idx.size)

    def __getitem__(self, i):
        """
        Build one (x, x_mark, y) triple from cached start index.
        NaN/inf safeguards applied again just in case (paranoia).
        """
        s = int(self.idx[i])      # start
        e = s + self.L            # end of input window
        p = e + self.H            # end of prediction window

        x = self.X[s:e, :]        # [L, D]
        y = self.X[e:p, -1:]      # [H, 1] last column = target
        x_mark = np.zeros((self.L, 0), dtype=np.float32)  # placeholder for time features

        # Extra safety replacement (should already be finite from indexing stage)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(x), torch.from_numpy(x_mark), torch.from_numpy(y)

    @staticmethod
    def load_matrix_from_csv(path: str, skip_header: int = 1) -> np.ndarray:
        """
        Read CSV, keep numeric columns only, return float32 matrix [T, D].
        If only one numeric column exists, ensure shape is [T, 1].

        Notes:
          * Non-numeric columns (timestamps, strings) are dropped via pandas dtype selection.
        """
        import pandas as pd
        df = pd.read_csv(path)
        num = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if num.ndim == 1:
            num = num.reshape(-1, 1)
        return num


# ---------------------------
# Dataloaders builder
# ---------------------------
def build_dataloaders(csv_path: str,
                      seq_len: int,
                      pred_len: int,
                      batch: int,
                      workers: int = 2,
                      skip_header: int = 1,
                      pin_memory: bool = True,
                      verbose: bool = False,
                      seed: int = 42):
    """
    Build train/val/test DataLoaders with standardized inputs.

    Steps:
      1) Load CSV -> numeric matrix Xraw [T, D].
      2) Clean X (finite-only via causal forward-fill; leading NaNs -> 0).
      3) Fit mu/sd on the first 70% of rows by time (min rows = L+H).
      4) Apply scaling to full X.
      5) Create SlidingWindowDataset on X_scaled.
      6) Split windows in time order into train/val/test with a leakage gap = H.
      7) Return DataLoaders and scaling tensors (mu, sd, mu_t, sd_t).

    Returns:
      dl_tr, dl_va, dl_te,
      mu (1D tensor[D]), sd (1D tensor[D]),
      mu_t (tensor[]), sd_t (tensor[])  # target channel stats for inverse-scaling

    Notes on splitting:
      * Splits are in "window index" space (after filtering invalid windows),
        preserving temporal order.
      * I inserted a gap of size H between train→val and val→test to avoid
        leakage from overlapping windows near boundaries.
    """
    # 1) Load numeric matrix
    Xraw = SlidingWindowDataset.load_matrix_from_csv(csv_path, skip_header=skip_header)
    # 2) Clean: causal forward-fill NaNs per column, then zero any remaining non-finite
    X = _clean_matrix(Xraw)

    # 3) Compute scaling stats on the first 70% of rows (by time),
    #    but ensure at least seq_len + pred_len rows are available for robust stats.
    T = X.shape[0]
    tr_rows_end = int(T * 0.7)
    tr_rows_end = max(tr_rows_end, seq_len + pred_len)
    tr_rows_end = min(tr_rows_end, T)

    mu, sd = _fit_scale(X[:tr_rows_end])
    mu_t = mu[-1]  # target channel mean (last column)
    sd_t = sd[-1]  # target channel std  (last column)

    # 4) Apply scaling to all rows using frozen training stats
    X_scaled = _apply_scale(X, mu, sd)

    # 5) Build dataset of valid sliding windows over standardized X
    ds_full = SlidingWindowDataset(X_scaled, seq_len, pred_len)
    N = len(ds_full)
    if N == 0:
        raise ValueError("No valid windows after cleaning. Check seq_len, pred_len, and CSV content.")

    # 6) Time-ordered split with leakage gaps (gap = pred_len)
    gap = pred_len
    i_tr_end   = int(N * 0.7)                           # ~70% windows for training
    i_va_start = min(i_tr_end + gap, N)                 # leave a gap after train
    i_va_end   = min(i_va_start + max(1, int(N * 0.1)), N)  # ~10% for validation
    i_te_start = min(i_va_end + gap, N)                 # gap before test (rest is test)

    # Build Subset views over the valid-start-index list
    idx = np.arange(N, dtype=np.int64)
    ds_tr = Subset(ds_full, idx[:i_tr_end])
    ds_va = Subset(ds_full, idx[i_va_start:i_va_end])
    ds_te = Subset(ds_full, idx[i_te_start:])

    # Deterministic DataLoader behavior across workers
    g = torch.Generator()
    g.manual_seed(42)

    # 7) DataLoaders:
    #    - shuffle only training (random order for optimization)
    #    - drop_last for training to keep batches consistent
    dl_tr = DataLoader(
        ds_tr, generator=g, batch_size=batch, shuffle=True,
        num_workers=workers, drop_last=True, pin_memory=pin_memory,
        worker_init_fn=_seed_worker
    )
    dl_va = DataLoader(
        ds_va, generator=g, batch_size=batch, shuffle=False,
        num_workers=workers, drop_last=False, pin_memory=pin_memory,
        worker_init_fn=_seed_worker
    )
    dl_te = DataLoader(
        ds_te, generator=g, batch_size=batch, shuffle=False,
        num_workers=workers, drop_last=False, pin_memory=pin_memory,
        worker_init_fn=_seed_worker
    )

    # Optional diagnostics
    if verbose:
        last = X[:, -1] if X.size else np.array([])
        finite_ratio = float(np.isfinite(last).mean()) if last.size else 0.0
        print("[DATA] path:", csv_path)
        print("[DATA] shape:", X.shape)
        print("[DATA] last-col finite ratio:", finite_ratio)
        print(f"[SPLITS] windows -> train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} gap={gap}")
        print(f"[SCALE] target mu={mu[-1]:.6f} sd={sd[-1]:.6f}")

    # Return scaling as torch tensors for convenient device transfers/broadcasting
    return dl_tr, dl_va, dl_te, torch.from_numpy(mu), torch.from_numpy(sd), torch.tensor(mu_t), torch.tensor(sd_t)
