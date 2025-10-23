# cli/train_deepar_gluonts.py
"""
Train and evaluate GluonTS DeepAR (PyTorch) on the last column of a CSV.

Key features:
- Auto-installs `requirements-deepar.txt` when running inside SageMaker.
- Reads the CSV (numeric columns only), uses the *last* column as the univariate target.
- Splits by time: train up to T-2H, validation up to T-H (GluonTS validates on its last H),
  and test on the final H.
- Trains DeepAR with Lightning CSV logger and EarlyStopping.
- Emits per-epoch `val_loss` by parsing Lightning's metrics.csv (for easy regex scraping).
- Computes MSE/MAE on validation and test via `make_evaluation_predictions`.
- Saves `metrics.json` and appends a one-line summary CSV; packages Lightning logs with the model.
"""

# ----------------------------
# Auto-install (SageMaker only)
# ----------------------------
import sys, subprocess, os
from pathlib import Path as _P
import glob, csv

# If we're on SageMaker, try to install a requirements file placed beside this script.
if os.environ.get("SM_TRAINING_ENV"):
    here = _P(__file__).parent
    req = here / "requirements-deepar.txt"
    print(f"[SETUP] Looking for {req}")
    if req.exists():
        print("[SETUP] Installing requirements-deepar.txt ...")
        # Install with no cache to reduce disk usage in constrained environments.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(req)])
    else:
        print("[SETUP] requirements-deepar.txt not found next to entry point.")

# Last-resort self-heal if gluonts still missing
try:
    import gluonts  # noqa: F401  # just testing import; not used yet
except Exception:
    # Install a known-good combo of GluonTS and Lightning that works together.
    print("[SETUP] gluonts not found; installing fallback (gluonts==0.14.4, lightning>=2.3,<2.6)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                           "gluonts==0.14.4", "lightning>=2.3,<2.6"])

# ----------------------------
# Now safe to import gluonts & friends
# ----------------------------
import argparse, json, numpy as np
from datetime import datetime

# Version-tolerant imports
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.torch.model.deepar import DeepAREstimator

# distributions: StudentT always exists; Gaussian can be Normal on some versions
from gluonts.torch.distributions import StudentTOutput
try:
    from gluonts.torch.distributions import GaussianOutput
except Exception:
    # Some versions expose NormalOutput instead of GaussianOutput
    from gluonts.torch.distributions import NormalOutput as GaussianOutput

# time features
try:
    from gluonts.time_feature import time_features_from_frequency_str
except Exception:
    # Older location fallback (unlikely on 0.14+, but harmless)
    from gluonts.time_feature import time_features_from_frequency_str

from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

# Cut a noisy GluonTS warning that doesn't affect correctness.
import warnings
warnings.filterwarnings("ignore", message=r"Using a non-tuple sequence", category=UserWarning, module=r"gluonts\.torch\.util")


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    """
    Define and parse command-line arguments. Defaults are conservative and
    work out-of-the-box for hourly data with a 96-step context/horizon.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)         # path to input CSV (numeric)
    ap.add_argument("--seq_len", type=int, default=96)             # context_length (L)
    ap.add_argument("--pred_len", type=int, default=96)            # prediction_length (H)
    ap.add_argument("--epochs", type=int, default=10)              # training epochs
    ap.add_argument("--batch_size", type=int, default=32)          # minibatch size
    ap.add_argument("--freq", type=str, default="h")               # pandas-style frequency string
    ap.add_argument("--seed", type=int, default=42)                # RNG seed
    ap.add_argument("--lr", type=float, default=1e-3)              # learning rate
    ap.add_argument("--weight_decay", type=float, default=0.0)     # optimizer weight decay
    ap.add_argument("--no_resume", type=str, default=None)         # reserved (not used internally)
    ap.add_argument("--resume_key", type=str, default=None)        # reserved (not used internally)
    ap.add_argument("--hidden_size", type=int, default=60)         # RNN hidden size
    ap.add_argument("--num_layers", type=int, default=2)           # number of RNN layers
    ap.add_argument("--dropout", type=float, default=0.1)          # dropout prob
    ap.add_argument("--likelihood", type=str, default="student_t", choices=["student_t","gaussian"])  # output distribution
    ap.add_argument("--scaling", type=str, default="zscore", choices=["zscore","none"])               # internal scaling

    # I/O
    ap.add_argument("--logdir", type=str, default="results")       # base output directory
    ap.add_argument("--run_name", type=str, default=None)          # optional run name tag
    ap.add_argument("--checkpoint_dir", type=str,
        default=os.environ.get("SM_CHECKPOINT_DIR") or "./results/spot_ckpts")  # for external tooling
    return ap.parse_args()

# ----------------------------
# Data loading helper
# ----------------------------
def load_target_last_column(csv_path):
    """
    Robustly load a CSV as a numeric matrix and return the *last column* as float32 target.
    Tries skipping a header line; falls back to no-skip.

    Assumptions:
    - CSV is numeric-only (non-numeric tokens will cause NaNs).
    - If single-column, reshape to (N, 1) so slicing by last column works uniformly.
    """
    for skip in (1, 0):
        try:
            X = np.genfromtxt(csv_path, delimiter=",", skip_header=skip)
            break
        except Exception:
            X = None
    if X is None or X.size == 0:
        raise RuntimeError(f"Could not read CSV: {csv_path}")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X[:, -1].astype(np.float32)  # last column as target

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    # Ensure output folders exist.
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Seeds (NumPy + optional torch/random)
    np.random.seed(args.seed)
    try:
        import torch, random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        # If torch isn't importable here, just proceed (handled later).
        pass

    # Load target series from the last CSV column
    target = load_target_last_column(args.csv_path)
    T = len(target)                         # total length
    L, H = args.seq_len, args.pred_len     # context/horizon

    # Fit μ,σ on first 70% of target (min L+H), same policy as PyTorch pipeline
    train_rows_end = max(int(0.7 * T), L + H)   # ensure enough rows to estimate stats
    train_rows_end = min(train_rows_end, T)     # cap within bounds
    mu_t = float(target[:train_rows_end].mean())
    sd_t = float(target[:train_rows_end].std())
    if sd_t < 1e-6:
        sd_t = 1.0  # avoid divide-by-zero downstream

    # Guardrail: need at least L + 2H points for train/val/test slicing below
    if T <= L + 2 * H:
        raise ValueError(f"Not enough points (T={T}) for context {L} and horizon {H}.")

    # Split series by time:
    # - train: up to T-2H
    # - val:   up to T-H (GluonTS validates on that series' last H)
    # - test:  full series; evaluate on final H
    train_series = target[: T - 2 * H]
    val_series   = target[: T - 1 * H]   # GluonTS validates on its last H
    test_series  = target                 # evaluate on final H

    # GluonTS datasets (univariate)
    freq = args.freq.lower()
    name = os.path.basename(args.csv_path).lower()
    # Small heuristics to auto-pick a better freq for some known datasets by name.
    if "weather" in name:
        freq = "10min"
    elif "ett" in name:
        freq = "h"
    elif "smd" in name:
        freq = "s"
    start_ts = "2020-01-01 00:00:00"  # fixed anchor timestamp; absolute value isn't used in metrics
    train_ds = ListDataset([{"start": start_ts, "target": train_series}], freq=freq)
    val_ds   = ListDataset([{"start": start_ts, "target": val_series  }], freq=freq)
    test_ds  = ListDataset([{"start": start_ts, "target": test_series }], freq=freq)

    # Lightning CSV logger: writes metrics.csv under logdir/name/version_*/
    logger = CSVLogger(save_dir=args.logdir, name="deepar_runs")

    # Estimator config:
    # Keep it simple & stable. Force CPU locally to avoid MPS gamma op gaps.
    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False

    import inspect
    time_feats = time_features_from_frequency_str(freq)                 # calendar covariates
    distr = StudentTOutput() if args.likelihood == "student_t" else GaussianOutput()
    use_scaling = (args.scaling == "zscore")                            # built-in scaling toggle

    # Collect kwargs; some keys may not exist depending on GluonTS version.
    deepar_kwargs = dict(
        freq=freq,
        prediction_length=H,
        context_length=L,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_batches_per_epoch=50,               # cap steps per epoch for speed/stability
        distr_output=distr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,                   # may or may not be supported
        scaling=use_scaling,
        use_feat_dynamic_real=True,             # enable time_features as dynamic real covariates
        time_features=time_feats,
        trainer_kwargs={
            "max_epochs": args.epochs,
            "enable_progress_bar": False,       # quieter logs (CSV captures metrics)
            "logger": logger,
            "check_val_every_n_epoch": 1,
            "log_every_n_steps": 1,
            "accelerator": "gpu" if has_torch and torch.cuda.is_available() else "cpu",
            "devices": 1,
            "callbacks": [EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        },
    )

    # Filter kwargs by the installed signature; remap dropout→dropout_rate if needed
    sig = inspect.signature(DeepAREstimator.__init__)
    params = set(sig.parameters.keys())
    if "dropout" in deepar_kwargs and "dropout" not in params and "dropout_rate" in params:
        # Some versions of GluonTS expect dropout_rate instead of dropout
        deepar_kwargs["dropout_rate"] = deepar_kwargs.pop("dropout")

    # Only pass arguments supported by the installed GluonTS version
    safe_kwargs = {k: v for k, v in deepar_kwargs.items() if k in params}
    est = DeepAREstimator(**safe_kwargs)

    # Train → returns a Predictor
    predictor = est.train(train_ds, validation_data=val_ds)

    # -----------------------------------------------------------------------
    # Emit per-epoch val_loss (and a derived "nll") from Lightning metrics:
    # This parses the latest 'metrics.csv' so upstream tools can regex it.
    # -----------------------------------------------------------------------
    from pathlib import Path
    runs_dir = Path(args.logdir) / "deepar_runs"
    try:
        latest = sorted(runs_dir.glob("version_*"))[-1]  # pick the most recent run folder
        mpath = latest / "metrics.csv"

        per_epoch = {}  # epoch -> (step, val_loss)
        with open(mpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("val_loss")
                if not v:
                    continue
                ep = int(float(row["epoch"]))   # 0-based in file
                step = int(float(row["step"]))
                val = float(v)
                # Keep the last (largest step) val_loss per epoch
                if ep not in per_epoch or step >= per_epoch[ep][0]:
                    per_epoch[ep] = (step, val)

        dataset_name = os.path.basename(args.csv_path).split(".")[0]
        for ep in sorted(per_epoch.keys()):
            val = per_epoch[ep][1]
            # 1-based epoch in the printout; include dataset + horizon for your parser
            print(f"ep {ep+1:02d} | dataset {dataset_name} | horizon {H} | val_loss {val:.6f} | nll {-val:.6f}")

    except Exception as e:
        # Non-fatal: training already completed; this only affects nice-to-have logs.
        print(f"Warn: couldn't emit per-epoch val_loss from {runs_dir}: {e}")
    # -----------------------------------------------------------------------

    # ----------------------------
    # Validation metrics (MSE/MAE)
    # ----------------------------
    # GluonTS helper returns iterators; we pull the first (and only) item for univariate case.
    forecast_it_v, ts_it_v = make_evaluation_predictions(val_ds, predictor, num_samples=100)
    val_fc = list(forecast_it_v)[0].mean           # mean forecast over samples, length H
    val_gt = list(ts_it_v)[0].values[-H:]          # ground truth last H points
    val_mse = float(np.mean((val_fc - val_gt) ** 2))
    val_mae = float(np.mean(np.abs(val_fc - val_gt)))

    # ----------------------------
    # Test metrics (MSE/MAE)
    # ----------------------------
    forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor, num_samples=100)
    forecasts = list(forecast_it)
    series = list(ts_it)
    fc_mean = forecasts[0].mean  # [H] forecast mean
    gt = series[0].values[-H:]   # [H] ground truth
    mse = float(np.mean((fc_mean - gt) ** 2))
    mae = float(np.mean(np.abs(fc_mean - gt)))

    # Normalized (z-space) metrics to match PyTorch "mse/mae"
    # (Use sd_t estimated from early segment to avoid leakage.)
    val_mse_z = float(np.mean(((val_fc - val_gt) / sd_t) ** 2))
    val_mae_z = float(np.mean(np.abs((val_fc - val_gt) / sd_t)))

    mse_z = float(np.mean(((fc_mean - gt) / sd_t) ** 2))
    mae_z = float(np.mean(np.abs((fc_mean - gt) / sd_t)))

    # ----- naive last-value baseline (original scale) -----
    # Proper naive = repeat last observed value for all H:
    # last observed value before each horizon
    val_anchor  = val_series[-H-1]   # last point before the validation horizon
    test_anchor = test_series[-H-1]  # last point before the test horizon

    val_last  = np.full_like(val_gt, val_anchor)
    test_last = np.full_like(gt,      test_anchor)

    # Baseline metrics (original scale)
    naive_val_mse  = float(np.mean((val_last  - val_gt) ** 2))
    naive_val_mae  = float(np.mean(np.abs(val_last  - val_gt)))
    naive_test_mse = float(np.mean((test_last - gt) ** 2))
    naive_test_mae = float(np.mean(np.abs(test_last - gt)))

    # z-space versions (divide by sd_t)
    naive_val_mse_z  = float(np.mean(((val_last  - val_gt) / sd_t) ** 2))
    naive_val_mae_z  = float(np.mean(np.abs((val_last  - val_gt) / sd_t)))
    naive_test_mse_z = float(np.mean(((test_last - gt) / sd_t) ** 2))
    naive_test_mae_z = float(np.mean(np.abs((test_last - gt) / sd_t)))

    # ----------------------------
    # Persist metrics + logs
    # ----------------------------
    dataset_name = os.path.basename(args.csv_path).split(".")[0]
    # If no run_name given, synthesize a readable one with dataset/L/H and timestamp.
    run_name = args.run_name or f"deepar_official_{dataset_name}_L{L}_H{H}_{datetime.now().strftime('%m%d-%H%M')}"

    # On SageMaker, prefer SM_MODEL_DIR; locally, write under logdir/model.
    model_dir = os.environ.get("SM_MODEL_DIR", os.path.join(args.logdir, "model"))
    os.makedirs(model_dir, exist_ok=True)
    # Dump metrics.json with both normalized and original scale metrics, plus baseline.
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({
            "model": "deepar_official",
            "dataset": dataset_name,
            "seq_len": L,
            "pred_len": H,
            "epochs": args.epochs,

            # z-space (normalized) — comparable to PyTorch "mse/mae"
            "val_mse": val_mse_z,
            "val_mae": val_mae_z,
            "mse": mse_z,
            "mae": mae_z,

            # original scale — comparable to PyTorch "mse_inv/mae_inv"
            "val_mse_inv": val_mse,
            "val_mae_inv": val_mae,
            "mse_inv": mse,
            "mae_inv": mae,

            # for traceability
            "mu_t": mu_t, "sd_t": sd_t,

            # naive baseline (normalized + original)
            "naive_val_mse": naive_val_mse_z,
            "naive_val_mae": naive_val_mae_z,
            "naive_mse": naive_test_mse_z,
            "naive_mae": naive_test_mae_z,
            "naive_val_mse_inv": naive_val_mse,
            "naive_val_mae_inv": naive_val_mae,
            "naive_mse_inv": naive_test_mse,
            "naive_mae_inv": naive_test_mae,
        }, f, indent=2)

    # Package Lightning CSV logs with the model for convenience
    try:
        import shutil, glob
        runs_dir = os.path.join(args.logdir, "deepar_runs")
        latest = sorted(glob.glob(os.path.join(runs_dir, "version_*")))[-1]
        shutil.copytree(latest, os.path.join(model_dir, "deepar_logs"), dirs_exist_ok=True)
    except Exception as e:
        # Best-effort; lack of logs in model dir isn't fatal.
        print("Warn: could not package CSV logs:", e)

    # Append a one-line summary CSV for easy experiment tracking across runs.
    out_csv = os.path.join(args.logdir, "deepar_official.csv")
    header = ["model","dataset","seq_len","pred_len","seed",
            "val_mse","val_mae","mse","mae",
            "val_mse_inv","val_mae_inv","mse_inv","mae_inv",
            "lr","batch","epochs","params", "naive_val_mse","naive_val_mae","naive_mse","naive_mae",
"naive_val_mse_inv","naive_val_mae_inv","naive_mse_inv","naive_mae_inv",]
    row = ["deepar_official", dataset_name, str(L), str(H), str(args.seed),
        f"{val_mse_z:.6f}", f"{val_mae_z:.6f}", f"{mse_z:.6f}", f"{mae_z:.6f}",
        f"{val_mse:.6f}",  f"{val_mae:.6f}",  f"{mse:.6f}",  f"{mae:.6f}",
        str(args.lr), str(args.batch_size), str(args.epochs), "n/a", f"{naive_val_mse_z:.6f}", f"{naive_val_mae_z:.6f}", f"{naive_test_mse_z:.6f}", f"{naive_test_mae_z:.6f}",
f"{naive_val_mse:.6f}", f"{naive_val_mae:.6f}", f"{naive_test_mse:.6f}", f"{naive_test_mae:.6f}"]

    # Create the CSV if missing; otherwise append one row.
    exists = os.path.exists(out_csv)
    with open(out_csv, "a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(row) + "\n")

    # Console summary (kept the same)
    print(f"val_mse={val_mse:.6f}")
    print(f"[VAL] mse {val_mse:.4f} | mae {val_mae:.4f}")
    print(f"[TEST] mse {mse:.4f} | mae {mae:.4f}")
    print(f"[VAL z] mse {val_mse_z:.4f} | mae {val_mae_z:.4f}")
    print(f"[TEST z] mse {mse_z:.4f} | mae {mae_z:.4f}")


# ----------------------------
# CLI entrypoint
# ----------------------------
if __name__ == "__main__":
    main()
