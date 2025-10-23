from __future__ import annotations
import time, json, os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless jobs (e.g., SageMaker); must be set before pyplot import
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")  # keep original styling (explicitly pinned versioned style)

# ---------------------------
# Filesystem helpers
# ---------------------------
def _ensure_dir(p: Path) -> Path:
    """
    Ensure a directory exists and return it.

    Behavior:
      - Creates `p` and any missing parents (no error if it already exists).
      - Returns the same Path object for chaining.

    Args:
      p: Path to a directory to create/ensure.

    Returns:
      Path: the same path, guaranteed to exist as a directory.
    """
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------
# Gradient diagnostics
# ---------------------------
def global_grad_norm(model) -> float:
    """
    Compute the global L2 norm of gradients across all model parameters.

    Notes:
      - Parameters with `grad is None` are skipped (e.g., frozen layers).
      - If no parameter has a gradient, returns 0.0.

    Args:
      model: torch.nn.Module with parameters potentially holding `.grad`.

    Returns:
      float: sqrt(sum_i ||g_i||_2^2) aggregated over all params.
    """
    import torch  # local import to avoid hard dependency when not used
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # p.grad.data.norm(2) returns the L2 norm; square then accumulate
            total += float(p.grad.data.norm(2).item() ** 2)
    return total ** 0.5


# ---------------------------
# Per-epoch logging schema
# ---------------------------
@dataclass
class EpochRow:
    """
    Row schema for epoch-wise logging. All fields match the training loop payload.

    Required fields:
      - epoch, train_loss, val_mse, val_mae, lr, grad_norm, time_sec

    Optional fields (default None):
      - val_mse_inv, val_mae_inv, val_rse

    Rationale:
      - Keeping optional metrics as None preserves forward/backward compatibility
        with existing CSV/JSON consumers.
    """
    epoch: int
    train_loss: float
    val_mse: float
    val_mae: float
    val_mse_inv: float | None = None
    val_mae_inv: float | None = None
    lr: float = 0.0
    grad_norm: float = 0.0
    time_sec: float = 0.0
    val_rse: float | None = None


# ---------------------------
# Tracker: logs, plots, summaries, forecast dumps
# ---------------------------
class MetricsTracker:
    """
    Tracks training progress and artifacts for a single experiment/run.

    Responsibilities:
      - Persist run hyperparameters (`hparams.json`).
      - Append per-epoch metrics to `epoch_log.csv` (and keep an in-memory list).
      - Write `summary.json` with best/final metrics and total time.
      - Generate diagnostic plots under `figs/`:
          * convergence.png (train loss & val MSE [+ val_mse_inv if available])
          * lr_schedule.png
          * grad_norm.png
      - Optionally dump forecast pairs under `preds/` as CSV + PNG.

    Directory layout (created under `out_dir/run_name/`):
      hparams.json
      epoch_log.csv
      summary.json
      figs/
        convergence.png
        lr_schedule.png
        grad_norm.png
      preds/
        <tag>.csv
        <tag>.png
    """

    def __init__(self, out_dir: str | Path, run_name: str, hparams: Optional[Dict[str, Any]] = None):
        """
        Initialize tracker directories and persist hyperparameters.

        Args:
          out_dir: Root output directory (e.g., "results").
          run_name: Subdirectory name for this run (e.g., "model_dataset_timestamp").
          hparams: Arbitrary hyperparameters/config dict to save for provenance.
        """
        # Resolve and create directories
        self.out_dir = _ensure_dir(Path(out_dir))
        self.run_dir = _ensure_dir(self.out_dir / run_name)
        self.fig_dir = _ensure_dir(self.run_dir / "figs")
        self.pred_dir = _ensure_dir(self.run_dir / "preds")

        # In-memory epoch rows and preserved hyperparameters
        self.rows: List[EpochRow] = []
        self.hparams = hparams or {}

        # Epoch timing (start time; None when not timing)
        self._t0 = None

        # Persist hyperparameters immediately (human-readable JSON)
        (self.run_dir / "hparams.json").write_text(json.dumps(self.hparams, indent=2))

    # -----------------------
    # Timing helpers
    # -----------------------
    def epoch_start(self):
        """
        Mark the start time for an epoch. Must be paired with `epoch_end`.
        """
        self._t0 = time.time()

    def epoch_end(self) -> float:
        """
        End the epoch timer and return elapsed seconds since `epoch_start`.

        Returns:
          float: elapsed seconds for the epoch.

        Raises:
          AssertionError: if called before `epoch_start`.
        """
        assert self._t0 is not None, "epoch_start has not been called"
        dt = time.time() - self._t0
        self._t0 = None
        return dt

    # -----------------------
    # Logging
    # -----------------------
    def log_epoch(self, **kwargs):
        """
        Append a single epoch row to the in-memory list and to `epoch_log.csv`.

        Required kwargs:
          epoch, train_loss, val_mse, val_mae, lr, grad_norm, time_sec

        Optional kwargs (if present will be stored as well):
          val_mse_inv, val_mae_inv, val_rse

        Behavior:
          - Writes/append to CSV with header on first write.
          - Maintains in-memory list for later summarization/plotting.
        """
        # Align kwargs to dataclass fields for stability across code versions
        row_dict = {f: kwargs.get(f, None) for f in EpochRow.__dataclass_fields__.keys()}

        # Validate presence of required fields (explicit error helps debugging)
        for req in ["epoch", "train_loss", "val_mse", "val_mae", "lr", "grad_norm", "time_sec"]:
            if row_dict[req] is None:
                raise ValueError(f"log_epoch missing required field: {req}")

        # Create dataclass instance (type: ignore keeps mypy happy with optional fields)
        row = EpochRow(**row_dict)  # type: ignore
        self.rows.append(row)

        # Append to CSV (header only if file doesn't exist yet)
        df = pd.DataFrame([asdict(self.rows[-1])])
        csv_path = self.run_dir / "epoch_log.csv"
        header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", header=header, index=False)

    def log_forecast(self, y_true, y_pred, tag: str):
        """
        Save a single forecast pair (true vs pred) as CSV and render a small PNG line plot.

        Args:
          y_true: 1D array-like of ground-truth values for horizon H.
          y_pred: 1D array-like of predicted values for horizon H.
          tag:    identifier used as filename stem (e.g., "val_ep10_last_horizon").
        """
        import numpy as np
        y_true = np.asarray(y_true).astype(float)
        y_pred = np.asarray(y_pred).astype(float)

        # Persist CSV (two columns: y_true, y_pred)
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        df.to_csv(self.pred_dir / f"{tag}.csv", index=False)

        # Quick line plot; minimal styling for clarity
        plt.figure(figsize=(8, 3))
        plt.plot(y_true, label="true")
        plt.plot(y_pred, label="pred")
        plt.title(tag)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.fig_dir / f"{tag}.png", dpi=200)
        plt.close()

    # -----------------------
    # Finalization & summaries
    # -----------------------
    def finalize(self):
        """
        Finalize the run:
          - Rewrite the full `epoch_log.csv` deterministically sorted by epoch.
          - Compute and save `summary.json`:
              * best_val_mse and its epoch,
              * final_val_mse / final_val_mae,
              * total_time_sec,
              * hparams snapshot.
          - Generate plots (convergence, lr schedule, grad norm) if columns exist.
          - Print a concise summary line.

        Notes:
          - Safely handles missing values (e.g., if some optional metrics are None).
          - If no rows logged, this is a no-op.
        """
        if not self.rows:
            return

        # Write full epoch log deterministically sorted by epoch
        df = pd.DataFrame([asdict(r) for r in self.rows]).sort_values("epoch")
        df.to_csv(self.run_dir / "epoch_log.csv", index=False)

        # Compute best/final aggregates (robust to missing values)
        df_mse = df.dropna(subset=["val_mse"]) if "val_mse" in df.columns else pd.DataFrame()
        if len(df_mse) == 0:
            best_val_mse, best_epoch = None, None
        else:
            best_idx = df_mse["val_mse"].idxmin()
            best_val_mse = float(df_mse.loc[best_idx, "val_mse"])
            best_epoch = int(df_mse.loc[best_idx, "epoch"])

        final_val_mse = float(df["val_mse"].iloc[-1]) if pd.notna(df["val_mse"].iloc[-1]) else None
        final_val_mae = float(df["val_mae"].iloc[-1]) if pd.notna(df["val_mae"].iloc[-1]) else None

        summary = {
            "best_val_mse": best_val_mse,
            "best_epoch": best_epoch,
            "final_val_mse": final_val_mse,
            "final_val_mae": final_val_mae,
            "total_time_sec": float(df["time_sec"].sum()) if "time_sec" in df.columns else None,
            "hparams": self.hparams,
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # Forward-fill missing values (e.g., warmup epochs) for nicer continuous plots
        df_ff = df.ffill()

        # Convergence plot (train_loss vs val_mse [+ val_mse_inv if available])
        if {"epoch", "train_loss", "val_mse"}.issubset(df_ff.columns) and (
            df_ff["train_loss"].notna().any() or df_ff["val_mse"].notna().any()
        ):
            self._plot_convergence(df_ff)

        # LR schedule plot
        if "lr" in df_ff.columns and df_ff["lr"].notna().any():
            self._plot_lr(df_ff)

        # Grad norm plot
        if "grad_norm" in df_ff.columns and df_ff["grad_norm"].notna().any():
            self._plot_gradnorm(df_ff)

        # Final console summary (prints 'None' if best/final are unavailable)
        print(
            f"[Tracker] best_val_mse={best_val_mse:.6f} at epoch {best_epoch}, "
            f"final_val_mse={final_val_mse:.6f}, total_time={summary['total_time_sec']:.1f}s"
        )

    # -----------------------
    # Plot helpers
    # -----------------------
    def _plot_convergence(self, df: pd.DataFrame):
        """
        Plot train loss vs validation MSE (and optional inverse MSE) over epochs.

        Saves:
          figs/convergence.png
        """
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_mse"], label="val_mse")
        if "val_mse_inv" in df.columns and df["val_mse_inv"].notna().any():
            plt.plot(df["epoch"], df["val_mse_inv"], "--", label="val_mse_inv")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.fig_dir / "convergence.png", dpi=200)
        plt.close()

    def _plot_lr(self, df: pd.DataFrame):
        """
        Plot the learning rate (per epoch). If using per-iteration schedulers,
        this curve reflects end-of-epoch values logged by the caller.

        Saves:
          figs/lr_schedule.png
        """
        plt.figure(figsize=(7, 3))
        plt.plot(df["epoch"], df["lr"])
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "lr_schedule.png", dpi=200)
        plt.close()

    def _plot_gradnorm(self, df: pd.DataFrame):
        """
        Plot global gradient norm across epochs.

        Saves:
          figs/grad_norm.png
        """
        plt.figure(figsize=(7, 3))
        plt.plot(df["epoch"], df["grad_norm"])
        plt.xlabel("epoch")
        plt.ylabel("grad_norm")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "grad_norm.png", dpi=200)
        plt.close()
