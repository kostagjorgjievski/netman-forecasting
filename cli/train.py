# cli/train.py
"""
Training script used by SageMaker (and locally) to train/evaluate
time-series models (iTransformer, PatchTST, DeepAR).

Key features:
- Auto-installs requirements when running inside SageMaker.
- Builds model/dataloaders, trains with AdamW + ReduceLROnPlateau.
- Tracks metrics via MetricsTracker and supports early stopping.
- Saves best checkpoint and exports artifacts/metrics to SM_MODEL_DIR (SageMaker) or local.
- Provides several sanity/scale checks and a naive last-value baseline.
"""

# ----------------------------
# Imports (stdlib + third-party)
# ----------------------------
import argparse, os, math, torch
from datetime import datetime

# Project-specific modules
from src.models import build_model
from src.data.datasets import build_dataloaders
from src.train_eval import train_epoch, evaluate  # evaluate must accept mu_sd
from src.utils import set_seed, auto_device, count_params, csv_log

# Third-party + torch utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
import json, shutil
from src.metrics_tracker import MetricsTracker, global_grad_norm
import time
from tqdm import tqdm
import shutil  # (duplicated import left intact to keep behavior identical)
from torch.optim.lr_scheduler import OneCycleLR

# Auto-install helpers (used only in SageMaker)
# (duplicated comment preserved to keep original structure)
# Auto-install only inside SageMaker containers
import sys, subprocess
from pathlib import Path as _P
import pandas as pd

# ----------------------------
# Forward helper used for eval-mode parity checks
# ----------------------------
def _forward_evalmode(model, x, xmark, device):
    """Run a forward pass with .eval() behavior, then restore original mode."""
    was_training = model.training
    model.eval()  # use eval behavior (BN running stats, dropout off)
    # Forward signature matches project convention: (x, xmark, dec_inp, dec_mark)
    out = model(x, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
    model.train(was_training)  # restore training flag for optimizer, etc.
    # Some wrappers return (out, aux); normalize to tensor
    return out[0] if isinstance(out, (list, tuple)) else out


# ----------------------------
# Diagnostics around normalization layers and dropout
# ----------------------------
def count_norms(model):
    """Print how many BatchNorm/LayerNorm layers exist in the model."""
    from collections import Counter
    c = Counter(type(m).__name__ for m in model.modules())
    # Count common BN/LayerNorm types across 1D/2D/3D variants
    bn = sum(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules())
    ln = sum(isinstance(m, nn.LayerNorm) for m in model.modules())
    print(f"[norms] BatchNorm={bn} LayerNorm={ln}")

def freeze_batchnorm(model):
    """Freeze BatchNorm layers to eval mode and make their params non-trainable."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()                       # lock running stats; disable batch-stat updates
            for p in m.parameters():
                p.requires_grad = False    # no updates via optimizer
            n += 1
    print(f"[norms] Froze {n} BatchNorm layers to eval mode")

def zero_all_dropout(model):
    """Set all Dropout layers' probability to zero for deterministic behavior."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.p = 0.0
            n += 1
    print(f"[dropout] zeroed {n} Dropout layers (all variants)")


# ----------------------------
# Naive baseline (last value) in scaled space (+ optional inverse metrics)
# ----------------------------
def naive_last_value(loader, device, mu_sd=None):
    """
    Last-value baseline computed in scaled space for fairness.
    If mu_sd is provided, also returns inverse-scaled metrics.

    Args:
        loader: PyTorch DataLoader yielding (x, xmark, y) with scaled tensors.
        device: torch.device to move tensors onto.
        mu_sd: Optional tuple (mu_t, sd_t) for inverse-scaling target channel.

    Returns:
        mse, mae, [mse_inv, mae_inv if mu_sd provided]
    """
    mse_f = nn.MSELoss(reduction="mean")
    mae_f = nn.L1Loss(reduction="mean")

    use_inv = mu_sd is not None
    if use_inv:
        mu_t, sd_t = mu_sd  # [1]-shaped tensors for the target channel

    sum_mse = 0.0
    sum_mae = 0.0
    sum_mse_inv = 0.0
    sum_mae_inv = 0.0
    n = 0

    with torch.no_grad():
        for x, _, y in loader:
            x = x.to(device)           # [B,L,D] scaled features (target is last channel)
            y = y.to(device)           # [B,H,1] scaled target

            last = x[:, -1:, -1:]      # [B,1,1] last observed target (scaled)
            pred = last.repeat(1, y.size(1), 1)  # repeat across horizon H

            bsz = x.size(0)
            sum_mse += float(mse_f(pred, y)) * bsz
            sum_mae += float(mae_f(pred, y)) * bsz

            if use_inv:
                # Inverse transform using scalar mu_t/sd_t for target channel
                pred_inv = pred * sd_t + mu_t
                y_inv    = y    * sd_t + mu_t
                sum_mse_inv += float(mse_f(pred_inv, y_inv)) * bsz
                sum_mae_inv += float(mae_f(pred_inv, y_inv)) * bsz

            n += bsz

    n = max(n, 1)  # avoid division by zero
    if use_inv:
        return sum_mse / n, sum_mae / n, sum_mse_inv / n, sum_mae_inv / n
    return sum_mse / n, sum_mae / n


# ----------------------------
# Global constants + CLI helpers
# ----------------------------
CODE_VERSION = "FINAL"

def str2bool(v):
    """Parse flexible boolean CLI flags (accepts 1/0, yes/no, true/false, on/off)."""
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y", "on")


# ----------------------------
# Environment-aware package auto-install (SageMaker only)
# ----------------------------
if os.environ.get("SM_TRAINING_ENV"):  # present on SageMaker
    base_req = _P(__file__).with_name("requirements.txt")
    if base_req.exists():
        # Install base requirements silently; helpful for ephemeral SM containers
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(base_req),
            "-qqq", "--disable-pip-version-check"
        ])

    # If launching DeepAR, pull its legacy pins as well (separate requirements file)
    if "--model" in sys.argv and "deepar" in " ".join(sys.argv):
        deepar_req = _P(__file__).with_name("requirements-deepar.txt")
        if deepar_req.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(deepar_req),
                "-qqq", "--disable-pip-version-check"
            ])


# ----------------------------
# Environment detection
# ----------------------------
def on_sagemaker() -> bool:
    """Return True when inside a SageMaker training container (env var or default SM config path)."""
    return bool(os.environ.get("SM_TRAINING_ENV")) or os.path.exists("/opt/ml/input/config/hyperparameters.json")


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    """Define and parse all CLI arguments (model, optimization, IO, etc.)."""
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--model", type=str, required=True,
                    choices=["itransformer", "patchtst", "deepar"])
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96,
                    choices=[96, 192, 336, 720])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    # model hyperparameters (unused ones can be ignored by specific models)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fc_dropout", type=float, default=0.0)     
    ap.add_argument("--head_dropout", type=float, default=0.0)  
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--embed", type=str, default="timeF")
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument("--use_norm", type=str2bool, default=False)
    ap.add_argument("--output_attention", type=str2bool, default=False)
    ap.add_argument("--last_horizon_only", type=str2bool, default=False,
                    help="If true, evaluate only the single final horizon (val/test), "
                         "to match DeepAR’s protocol.")

    # optimization
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # I/O
    ap.add_argument("--logdir", type=str, default="results")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--resume_key", type=str, default="")  # stable key across jobs for resume

    # PatchTST specific:
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--stride",    type=int, default=16)
    ap.add_argument("--revin", type=str2bool, default=True)
    ap.add_argument("--enc_in", type=int, default=None)

    # SageMaker-friendly paths (work locally too)
    sm = on_sagemaker()
    default_data_dir = os.environ.get("SM_CHANNEL_TRAINING") if sm else os.getcwd()
    default_ckpt_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints") if sm else "checkpoints"

    ap.add_argument("--data_dir", type=str, default=default_data_dir)
    ap.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir)
    ap.add_argument("--no_resume", type=str2bool, default=False)

    return ap.parse_args()


# ----------------------------
# Main training/evaluation routine
# ----------------------------
def main():
    """Entrypoint: build model/data, train with metrics tracking, export artifacts."""
    print("mode:", "sagemaker" if on_sagemaker() else "local")
    print("CODE_VERSION:", CODE_VERSION)

    args = parse_args()
    print("CONFIG:", vars(args))

    # Default run_name comes from the SageMaker job name if not provided by user
    sm_job = os.environ.get("SM_TRAINING_JOB_NAME")
    if args.run_name is None and sm_job:
        args.run_name = sm_job

    # Cap workers for small CPU instances to avoid dataloader oversubscription
    args.workers = min(args.workers, os.cpu_count() or 1)

    # Reproducibility + device selection (allow explicit "--device")
    set_seed(args.seed)
    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = auto_device()  # chooses cuda/mps/cpu based on availability

    # Lightweight config object that downstream constructors expect
    class Cfg: pass
    cfg = Cfg()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    # Some model builders expect this attribute; provide a default
    if not hasattr(cfg, "class_strategy"):
        cfg.class_strategy = "projection"

    # --- PatchTST config shim: infer enc_in (D) and fill defaults ---
    if args.model.lower() == "patchtst":
        # 1) Infer number of input channels if not provided
        if getattr(args, "enc_in", None) is not None:
            cfg.enc_in = int(args.enc_in)
        else:
            # Read CSV once to infer numeric column count
            df = pd.read_csv(args.csv_path)
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                raise ValueError("No numeric columns in CSV (drop timestamp etc.)")
            cfg.enc_in = int(num.shape[1])

        # 2) Fill other attributes accessed by the official wrapper
        defaults = dict(
            dropout=0.2,
            fc_dropout=0.2,
            head_dropout=0.0,
            attn_dropout=0.0,
            individual=False,
            patch_len=getattr(args, "patch_len", 16),
            stride=getattr(args, "stride", 16),
            padding_patch=None,
            revin=True,
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
            max_seq_len=1024,
            d_k=None, d_v=None,
            norm="LayerNorm",
            key_padding_mask="auto",
            padding_var=None,
            attn_mask=None,
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            pe="zeros",
            learn_pe=True,
            pretrain_head=False,
            head_type="flatten",
            verbose=False,
        )
        for k, v in defaults.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)
    # --- end PatchTST shim ---

    if hasattr(cfg, "enc_in"):
        print(f"[DEBUG] enc_in (channels) inferred from CSV = {cfg.enc_in}")

    # Build model according to args.model and move to device
    model = build_model(args.model, cfg).to(device)
    # Report number of normalization layers for debugging/troubleshooting
    count_norms(model)

    params = count_params(model)
    print(f"[{args.model}] device={device} params={params/1e6:.2f}M")

    # ------------------------
    # Dataloaders & scaling stats
    # ------------------------
    pin = (device.type == "cuda")  # pin_memory helps for GPU input pipelines
    dl_tr, dl_va, dl_te, mu, sd, mu_t, sd_t = build_dataloaders(
        csv_path=args.csv_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        pin_memory=pin,
    )

    # Optional: evaluate on the single final horizon only (DeepAR-style)
    if args.last_horizon_only:
        from torch.utils.data import DataLoader, Subset
        def _last_loader(dl):
            ds = dl.dataset
            last_idx = max(len(ds) - 1, 0)
            return DataLoader(
                Subset(ds, [last_idx]),
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=pin,
                collate_fn=getattr(dl, "collate_fn", None),
                drop_last=False
            )
        dl_va = _last_loader(dl_va)
        dl_te = _last_loader(dl_te)

    # Move per-feature mean/std and target mean/std to device (float32)
    mu   = mu.to(device).float()                 # [D]
    sd   = sd.to(device).float()                 # [D]
    mu_t = mu_t.to(device).float().reshape(1)    # [1] ensure broadcastable shape
    sd_t = sd_t.to(device).float().reshape(1)    # [1]

    # Quick validation std check in global-z space (sanity: ~1 if scaling OK)
    with torch.no_grad():
        import math
        tot = 0; sum2 = 0.0
        for xb, _, yb in dl_va:
            yb = yb.view(-1)  # [B*H]
            mask = torch.isfinite(yb)
            z = yb[mask]
            tot += z.numel()
            sum2 += float((z**2).sum())
        val_std_est = math.sqrt(sum2 / max(tot, 1))
        print(f"[DEBUG] estimated std of VALIDATION target in global-z space ≈ {val_std_est:.3f} (should be ~1 over train, ~1-ish on val)")

    # ------------------------
    # One-batch sanity checks for shapes/scale and train vs eval parity
    # ------------------------
    def batch_stats(tag, loader):
        """Print shapes, scaled stats, and single-batch MSE parity (eval vs train)."""
        xb, xmark, yb = next(iter(loader))
        xb, xmark, yb = xb.to(device), xmark.to(device), yb.to(device)

        # Eval-mode forward for deterministic behavior in diagnostics
        out = _forward_evalmode(model, xb, xmark, device)
        out_ot = out[..., -1:]  # ensure last-dim singleton for target channel

        print(f"[{tag}] shapes x={tuple(xb.shape)} y={tuple(yb.shape)}")
        print(f"[{tag}] y mean/std ~ {yb.mean():.4f} / {yb.std():.4f}")
        print(f"[{tag}] out raw shape = {tuple(out.shape)}")
        print(f"[{tag}] direct scaled MSE on 1 batch = {((out_ot - yb)**2).mean().item():.6f}")
        
        # Check train vs eval consistency for the same batch (BN/dropout effects)
        model.train()
        out_train = model(xb, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
        out_train = out_train[0] if isinstance(out_train, (list, tuple)) else out_train
        mse_train = ((out_train[..., -1:] - yb) ** 2).mean().item()
        print(f"[BATCH-STATS-PARITY] eval_mse={((out_ot - yb)**2).mean().item():.6f}, train_mse={mse_train:.6f}")

    print(f"[scale] mu_t={mu_t.item():.6f} sd_t={sd_t.item():.6f}")
    batch_stats("TRAIN", dl_tr)
    batch_stats("VAL",   dl_va)

    # ------------------------
    # Optimizer & LR scheduler setup
    # ------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Paths for checkpoints/artifacts (SageMaker + local)
    ckpt_root = os.path.join(args.logdir, "ckpts")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)  # for rolling trainer state (resume)

    dataset_name = os.path.basename(args.csv_path).split(".")[0]
    run_name = args.run_name or f"{args.model}_{dataset_name}_L{args.seq_len}_H{args.pred_len}_{datetime.now().strftime('%m%d-%H%M')}"
    best_model_path = os.path.join(ckpt_root, f"{run_name}.pth")  # best-epoch weights
    rolling_path = os.path.join(args.checkpoint_dir, f"{(args.resume_key or args.run_name or 'run')}.pt")  # spot/rolling state

    # Tracker manages per-epoch logging and directories for artifacts
    tracker = MetricsTracker(out_dir=args.logdir, run_name=run_name, hparams=vars(args))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )
    # Alternate LR schedule available; left here for experimentation:
    # train_steps = len(dl_tr)
    # scheduler = OneCycleLR(
    #     optimizer=opt,
    #     max_lr=args.lr,
    #     epochs=args.epochs,
    #     steps_per_epoch=train_steps,
    #     pct_start=0.3,
    # )

    # Early stopping configuration (manual loop with patience/min_delta)
    patience, waited = 5, 0
    min_delta = 1e-3
    history_rows = []  # compact CSV history saved at the end

    # ------------------------
    # Scale/parity check on a single batch (train vs eval)
    # ------------------------
    with torch.no_grad():
        xb, xmark, yb = next(iter(dl_tr))
        if hasattr(cfg, "enc_in"):
            # Sanity: channel count matches inferred enc_in for PatchTST
            assert xb.shape[2] == cfg.enc_in, f"enc_in={cfg.enc_in} but dataloader D={xb.shape[2]}"
        xb, xmark, yb = xb.to(device), xmark.to(device), yb.to(device)

        # forward in train-mode (model is currently in train mode)
        out_train = model(xb, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
        out_train = out_train[0] if isinstance(out_train, (list, tuple)) else out_train
        out_train = out_train[..., -1:]  # ensure [B,H,1]

        # forward in eval-mode (to check BN/Dropout differences)
        model.eval()
        out_eval = model(xb, xmark, torch.empty(0, device=device), torch.empty(0, device=device))
        out_eval = out_eval[0] if isinstance(out_eval, (list, tuple)) else out_eval
        out_eval = out_eval[..., -1:]
        model.train()

        # stats to inspect scaling behavior and parity
        x_target_std = xb[:, :, -1].std().item()  # std of the target channel in x
        print(
            "[SCALE-CHECK] x_target_std={:.6f} y_std={:.6f} "
            "out_train_std={:.6f} out_eval_std={:.6f}".format(
                x_target_std, yb.std().item(), out_train.std().item(), out_eval.std().item()
            )
        )
        print(
            "[SCALE-CHECK] x_target_mean={:.6f} y_mean={:.6f} "
            "out_train_mean={:.6f} out_eval_mean={:.6f}".format(
                xb[:, :, -1].mean().item(), yb.mean().item(),
                out_train.mean().item(), out_eval.mean().item()
            )
        )

        # quick MSEs on the same batch to detect eval/train drifts
        mse_train = ((out_train - yb) ** 2)[torch.isfinite(out_train) & torch.isfinite(yb)].mean().item()
        mse_eval  = ((out_eval  - yb) ** 2)[torch.isfinite(out_eval)  & torch.isfinite(yb)].mean().item()
        print(f"[SCALE-CHECK] mse_train={mse_train:.6f}  mse_eval={mse_eval:.6f}")
    # ----------------------------------------------------------------------

    # One more outside-of-pipeline sanity check on raw CSV last-column stats
    num = pd.read_csv(args.csv_path).select_dtypes(include=[np.number]).to_numpy()
    last_col = num[:, -1]
    print("OT mean:", np.mean(last_col), "std:", np.std(last_col))

    # ------------------------
    # Training loop with validation, diagnostics, and early stopping
    # ------------------------
    start_epoch = 1
    best = math.inf  # track best (lowest) validation MSE

    # Resume from rolling checkpoint if allowed and metadata matches
    if not args.no_resume and os.path.exists(rolling_path):
        try:
            ckpt = torch.load(rolling_path, map_location=device)
            same_job = (ckpt.get("run_name") == (args.run_name or ""))
            same_model = (ckpt.get("model") == args.model)
            if same_job and same_model:
                same_job = (ckpt.get("run_name") == (args.run_name or ""))
                same_model = (ckpt.get("model") == args.model)
                same_key = (args.resume_key and ckpt.get("resume_key") == args.resume_key)
                if (same_job or same_key) and same_model:
                    # Restore model/optimizer/scheduler and early-stopping state
                    model.load_state_dict(ckpt["model_state"])
                    opt.load_state_dict(ckpt["optim_state"])
                    scheduler.load_state_dict(ckpt["sched_state"])
                    best = ckpt.get("best", best)
                    waited = ckpt.get("waited", waited)
                    patience = ckpt.get("patience", patience)
                    min_delta = ckpt.get("min_delta", min_delta)
                    start_epoch = int(ckpt.get("epoch", 0)) + 1
                    print(f"[RESUME] Resumed at epoch {start_epoch} from {rolling_path} "
                      f"(run_name match={same_job}, resume_key match={same_key})")
            else:
                print("[RESUME] Checkpoint exists but run_name/model mismatch; starting fresh.")
        except Exception as e:
            print(f"[RESUME] Failed to load checkpoint ({e}); starting fresh.")

    # Main epoch loop
    for ep in range(start_epoch, args.epochs + 1):
        tracker.epoch_start()

        # One epoch of training (returns train MSE in scaled space)
        tr = train_epoch(model, dl_tr, opt, device)

        # Validation (scaled + inverse-scaled metrics)
        with torch.no_grad():
            vmse, vmae, vmse_inv, vmae_inv = evaluate(model, dl_va, device, mu_sd=(mu_t, sd_t))

        # --- Validation scale diagnostics in z-space ---
        with torch.no_grad():
            import math
            def rms(t): return (t.pow(2).mean().sqrt().item())
            ys = []
            for x, _, y in dl_va:
                ys.append(y.view(-1))
            y_val = torch.cat(ys)
            print("VAL mean(z):", y_val.mean().item(),
                "RMS(z):", rms(y_val),
                "STD(z):", y_val.std(unbiased=False).item())

            var_val = y_val.var(unbiased=False).item()
            rse = vmse / max(var_val, 1e-12)  # relative squared error vs var
            print(f"RSE (vmse/var_val): {rse:.6f}")
        # ------------------------------------

        # Naive last-value baseline on validation (scaled)
        with torch.no_grad():
            n_mse_va, n_mae_va = naive_last_value(dl_va, device)  # scaled space only
        print(f"[NAIVE-VAL] mse {n_mse_va:.6f} | mae {n_mae_va:.6f}")

        # Parity: training loss vs evaluate(model, dl_tr) on the same epoch
        with torch.no_grad():
            tr_eval, _ = evaluate(model, dl_tr, device)  # scaled
        print(f"[PARITY] train_epoch={tr:.6f}  evaluate_on_train={tr_eval:.6f}")

        # Log the epoch summary (printed + tracker JSON/CSV)
        lr = opt.param_groups[0]["lr"]
        print(f"ep {ep:02d} | train_mse {tr:.6f} | val_mse {vmse:.6f} | val_mae {vmae:.6f} | "
            f"val_mse_inv {vmse_inv:.6f} | val_mae_inv {vmae_inv:.6f} | lr={lr:.2e}")

        # Minimal history row (usable for quick plotting)
        history_rows.append({
            "epoch": ep, "train_mse": float(tr), "val_mse": float(vmse),
            "val_mae": float(vmae), "val_mse_inv": float(vmse_inv),
            "val_mae_inv": float(vmae_inv), "lr": float(lr), "rse": float(rse),
        })

        # Step LR scheduler on validation MSE
        scheduler.step(vmse)
        epoch_time = tracker.epoch_end()
        gn = global_grad_norm(model)

        # Persist metrics to tracker (keeps everything under run directory)
        tracker.log_epoch(
            epoch=ep,
            train_loss=tr,
            val_mse=vmse,
            val_mae=vmae,
            val_mse_inv=vmse_inv,
            val_mae_inv=vmae_inv,
            lr=lr,
            grad_norm=gn,
            time_sec=epoch_time,
            val_rse=rse,     # <- extra diagnostic metric
        )

        # --- rolling checkpoint for Spot resume (per epoch) ---
        ckpt_obj = {
            "run_name": args.run_name or "",
            "resume_key": args.resume_key or "",
            "model": args.model,
            "epoch": ep,
            "best": best,
            "waited": waited,
            "patience": patience,
            "min_delta": min_delta,
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "sched_state": scheduler.state_dict(),
        }
        tmp_path = rolling_path + ".tmp"
        torch.save(ckpt_obj, tmp_path)
        os.replace(tmp_path, rolling_path)  # atomic on POSIX
        # ------------------------------------------------------

        # Early stopping based on val MSE with min_delta
        if vmse < best - min_delta:
            best, waited = vmse, 0
            torch.save(model.state_dict(), best_model_path)  # stash best weights
        else: 
            waited += 1
            if waited >= patience:
                print("Early stop")
                break
        
    # ------------------------
    # Load best checkpoint and run final test
    # ------------------------
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"[INFO] Loaded best checkpoint for testing: {best_model_path}")

    with torch.no_grad():
        tmse, tmae, tmse_inv, tmae_inv = evaluate(model, dl_te, device, mu_sd=(mu_t, sd_t))
    print(f"[TEST] mse {tmse:.6f} | mae {tmae:.6f} | mse_inv {tmse_inv:.6f} | mae_inv {tmae_inv:.6f}")

    # Naive baseline on test (scaled + inverse)
    n_mse, n_mae, n_mse_inv, n_mae_inv = naive_last_value(dl_te, device, mu_sd=(mu_t, sd_t))
    print(f"[NAIVE-TEST] mse {n_mse:.6f} | mae {n_mae:.6f} | mse_inv {n_mse_inv:.6f} | mae_inv {n_mae_inv:.6f}")

    # Finalize tracker (flush/close artifacts)
    tracker.finalize()

    # ------------------------
    # Artifact export (SageMaker vs local)
    # ------------------------
    if on_sagemaker():
        model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    else:
        model_dir = Path(args.logdir) / "artifacts" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Copy tracker run directory next to model artifacts (for full provenance)
    from shutil import copytree
    art_dir = model_dir / "tracker"
    copytree(tracker.run_dir, art_dir, dirs_exist_ok=True)

    # Ensure logdir exists (for summary CSV below)
    os.makedirs(args.logdir, exist_ok=True)

    # One-line CSV summary (append/update) for quick experiments table
    header = ["model","dataset","seq_len","pred_len","seed","mse","mae","mse_inv","mae_inv","lr","batch","epochs","params"]
    row = [args.model, os.path.basename(args.csv_path), args.seq_len, args.pred_len,
        args.seed, f"{tmse:.6f}", f"{tmae:.6f}", f"{tmse_inv:.6f}", f"{tmae_inv:.6f}",
        args.lr, args.batch_size, args.epochs, params]

    csv_log(os.path.join(args.logdir, f"{args.model}.csv"), header, row)

    # Recompute model_dir choice (duplicated block preserved)
    if on_sagemaker():
        model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    else:
        model_dir = Path(args.logdir) / "artifacts" / run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save best weights if available; otherwise save the current model state
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, model_dir / "model.pt")
    else:
        torch.save(model.state_dict(), model_dir / "model.pt")

    # Persist metrics + config for downstream consumption (JSON contract)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump({
            "model": args.model,
            "dataset": os.path.basename(args.csv_path),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "epochs": args.epochs,
            "mse": float(tmse),
            "mae": float(tmae),
            "params": int(params),
            "eval_scope": "last_horizon" if args.last_horizon_only else "rolling"
        }, f, indent=2)

    # Write a tiny, stable training history file for plotting (one row per epoch)
    import csv as _csv
    hist_path = model_dir / "history.csv"
    if history_rows:
        with open(hist_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
            w.writeheader(); w.writerows(history_rows)

    # Run meta (helps Lambda/services know where to upload plots/results)
    with open(model_dir / "run_meta.json", "w") as f:
        json.dump({
            "run_name": args.run_name,
            "resume_key": args.resume_key,
            "output_s3": os.environ.get("SM_OUTPUT_DATA_DIR", ""),
            "checkpoint_dir": args.checkpoint_dir,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Also archive the one-line CSV next to the model for convenience
    shutil.copy2(os.path.join(args.logdir, f"{args.model}.csv"), model_dir / "results.csv")


# ----------------------------
# CLI entry point
# ----------------------------
if __name__ == "__main__":
    main()
