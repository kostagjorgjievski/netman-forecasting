"""
SageMaker training/HP-tuning launcher for time-series models (iTransformer, PatchTST, DeepAR).

- Builds a PyTorch SageMaker Estimator (or GluonTS entry point for DeepAR).
- Supports multiple forecast horizons (baseline runs) or single-horizon HPO.
- Uploads/reads code, data, checkpoints, and outputs to/from S3.

Usage (examples):
  python run.py --model itransformer --dataset ETT.csv --horizons 96 192 --epochs 30
  python run.py --model itransformer --tune 1 --horizons 96 --epochs 20

Notes:
- Region/bucket/role are configured below (placeholders here; fill with your own).
- Spot training is enabled with checkpointing on S3.
"""

# ----------------------------
# Standard library imports
# ----------------------------
import argparse, os, re, time
from pathlib import Path

# Third-party (AWS + SageMaker SDK)
import boto3, sagemaker
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

# ----------------------------
# Global configuration (AWS/S3/SageMaker session)
# ----------------------------
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Anonymized placeholders — replace with your own values before running.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
region = "<your-aws-region>"                       # e.g., "us-east-1"
bucket = "<your-s3-bucket-name>"                   # e.g., "my-forecasting-bucket"
ROLE = "<your-sagemaker-execution-role-arn>"       # e.g., "arn:aws:iam::123456789012:role/MySageMakerRole"

# Optional alternates (kept commented to mirror original structure)
# region = "<your-alt-aws-region>"
# bucket = "<your-alt-s3-bucket>"
# ROLE = "<your-alt-sagemaker-execution-role-arn>"

# Create a SageMaker session pinned to the chosen region.
sess = sagemaker.Session(boto_session=boto3.Session(region_name=region))

# S3 location where source code archives will be uploaded by SageMaker
code_loc = f"s3://{bucket}/code/"

# Channel mapping for training data. We use a "training" channel pointing
# at an S3 prefix that contains CSV files. FastFile enables streaming.
inputs = {"training": TrainingInput(f"s3://{bucket}/data/", input_mode="FastFile")}

# ----------------------------
# Utilities
# ----------------------------
def _mk_job_name(model: str, H: int, dataset: str) -> str:
    """
    Build a short, DNS-safe SageMaker job name.
    Includes '-last_horizon-h{H}' so runs are easily filterable in the console.

    Args:
        model: model string (itransformer|patchtst|deepar)
        H: forecast horizon
        dataset: dataset filename (used for context)

    Returns:
        A sanitized job name (<= 63 chars).
    """
    ds = Path(dataset).stem.lower()
    base = f"{model}-{ds}-last_horizon-h{H}-{int(time.time())}"
    # Normalize to lowercase letters, digits, and hyphens
    s = re.sub(r"[^a-z0-9-]", "-", base)
    # Collapse multiple hyphens and trim ends
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:63]  # SageMaker enforces a length limit

def _norm_ds_name(dataset: str) -> str:
    """Map dataset filename to a canonical key used by config helpers: ett|weather|smd|<custom>."""
    stem = Path(dataset).stem.lower()
    if "ett" in stem:
        return "ett"
    if "weather" in stem or "weather10" in stem or "wth" in stem:
        return "weather"
    if "smd" in stem:
        return "smd"
    # default to the stem if user passes a custom dataset name
    return stem

# ----------------------------
# Dataset- & model-specific configs
# ----------------------------
def cfg_patchtst(ds: str, H: int, smd_granularity: str):
    """
    Heuristic defaults for PatchTST, tailored by dataset key and horizon.
    Returns (seq_len, cfg_dict). cfg_dict merges into estimator hyperparameters.
    """
    ds = _norm_ds_name(ds)
    # Safe defaults
    cfg = dict(
        lr=1e-3, dropout=0.1, d_model=128, n_heads=8, e_layers=3, d_ff=256,
        batch_size=128, revin=True, fc_dropout=0.1,
        patch_len=24, stride=6,  # overridden per dataset below
    )

    if ds == "ett":  # hourly, period=24
        seq_len = 336 if H <= 336 else 672
        cfg.update(patch_len=24, stride=6, d_model=128, n_heads=8, e_layers=3, dropout=0.1, batch_size=128)

    elif ds == "weather":  # 10-min, daily period=144
        seq_len = 1008 if H <= 336 else 1440
        cfg.update(patch_len=144, stride=12, d_model=192, n_heads=8, e_layers=3, dropout=0.1, batch_size=64)

    elif ds == "smd":
        if smd_granularity == "second":
            seq_len = 3600 if H <= 336 else 5400
            cfg.update(patch_len=60, stride=10, d_model=192, n_heads=8, e_layers=3, dropout=0.2, batch_size=64)
        else:  # minute
            seq_len = (3600 if H <= 336 else 5400) // 60
            cfg.update(patch_len=60 // 1, stride=10 // 1, d_model=192, n_heads=8, e_layers=3, dropout=0.2, batch_size=64)
    else:
        seq_len = 336  # fallback if unknown

    return seq_len, cfg

def cfg_itransformer(ds: str, H: int, smd_granularity: str):
    """
    Heuristic defaults for iTransformer, tailored by dataset key and horizon.
    Returns (seq_len, cfg_dict). cfg_dict merges into estimator hyperparameters.
    """
    ds = _norm_ds_name(ds)
    # Safe defaults
    cfg = dict(lr=1e-3, dropout=0.1, d_model=128, n_heads=8, e_layers=2, d_ff=512, batch_size=128)

    if ds == "ett":  # hourly
        seq_len = 336 if H <= 336 else 504
        cfg.update(d_model=128, n_heads=8, e_layers=2, dropout=0.1, batch_size=128)

    elif ds == "weather":  # 10-min
        seq_len = 1008 if H <= 336 else 1440
        cfg.update(d_model=192, n_heads=8, e_layers=3, dropout=0.1, batch_size=64, d_ff=1024)

    elif ds == "smd":
        if smd_granularity == "second":
            # use the long end of the range for higher horizons
            seq_len = 3600 if H <= 336 else 5400
            cfg.update(d_model=128, n_heads=8, e_layers=3, dropout=0.3, batch_size=64, d_ff=1024)
        else:  # minute
            seq_len = (3600 if H <= 336 else 5400) // 60
            cfg.update(d_model=128, n_heads=8, e_layers=3, dropout=0.3, batch_size=64, d_ff=1024)
    else:
        seq_len = 336

    return seq_len, cfg

# ----------------------------
# Argparse
# ----------------------------
def parse_args():
    """
    CLI options:
      --model:         itransformer | patchtst | deepar
      --dataset:       CSV filename under the s3://.../data/ prefix
      --horizons:      one or more forecast horizons to run
      --epochs:        training epochs
      --instance:      instance type (e.g., ml.g4dn.xlarge)
      --enc_in:        optional channel count for multivariate models
      --smd_granularity: "second" or "minute" (SMD dataset sampling)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["itransformer", "patchtst", "deepar"], default="itransformer")
    ap.add_argument("--dataset", required=True, help="CSV filename under s3://.../data/")
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--epochs", type=int, default=50)  # default 50 epochs
    ap.add_argument("--instance", default="ml.g4dn.xlarge")
    ap.add_argument("--enc_in", type=int, default=None)
    ap.add_argument("--smd_granularity", choices=["second", "minute"], default="minute",
                    help="Sampling for SMD (minute = assignment text; second = earlier run).")
    return ap.parse_args()

def _freq_for(ds_key: str, smd_granularity: str) -> str:
    """
    Map canonical dataset key to pandas-style frequency for GluonTS DeepAR.
    """
    if ds_key == "ett":
        return "h"
    if ds_key == "weather":
        return "10min"  # GluonTS uses pandas offset aliases
    if ds_key == "smd":
        return "S" if smd_granularity == "second" else "min"
    return "h"  # sensible default

import math
def _deepar_ctx_len(ds_key: str, H: int, smd_granularity: str) -> int:
    """
    Choose a DeepAR context_length as max(1.5 * H, 3 * period), snapped to a multiple of period.
    Periods:
      ett: 24 (hourly)
      weather: 144 (10-minutely → 144 per day)
      smd: 60 (treat 1-min as 60 per hour)
    """
    if ds_key == "ett":
        period = 24
    elif ds_key == "weather":
        period = 144
    elif ds_key == "smd":
        period = 60
    else:
        period = 24

    base = max(int(1.5 * H), 3 * period)
    # Snap to an integer multiple of the period to help seasonality modeling
    ctx = int(math.ceil(base / period) * period)
    return ctx

def _deepar_hp_for(ds_key: str):
    """
    Heuristic DeepAR hyperparameters per dataset family.
    Returns a dict merged into estimator hyperparameters.
    """
    # Defaults target ETT-scale workloads
    hp = dict(hidden_size=60, num_layers=2, dropout=0.1, batch_size=128)
    if ds_key == "weather":
        hp.update(hidden_size=96, num_layers=2, dropout=0.1, batch_size=64)
    if ds_key == "smd":
        hp.update(hidden_size=96, num_layers=2, dropout=0.2, batch_size=64)
    return hp

# ----------------------------
# Estimator builder
# ----------------------------
def build_estimator(
    model: str,
    instance_type: str,
    csv_file: str,
    pred_len: int,
    epochs: int,
    enc_in: int | None,
    smd_granularity: str,
):
    """
    Construct a SageMaker Estimator with the right entry point and hyperparameters
    for the selected model, dataset, and horizon.
    """
    # Pull the correct AWS DLC image for PyTorch training in the given region/instance.
    img = image_uris.retrieve(
        framework="pytorch", region=region, version="2.4",
        py_version="py311", instance_type=instance_type, image_scope="training",
    )

    # Default to our PyTorch training entrypoint; DeepAR overrides this below.
    entry_point = "train.py"
    deps = ["src"]  # package project code alongside entry_point
    # Parse a simple scalar from training logs (used by SM metrics)
    metric_defs = [{"Name": "val_mse", "Regex": r"val_mse=([0-9.]+)"}]

    ds_key = _norm_ds_name(csv_file)

    # Base shared hyperparameters passed to entry_point. These map directly
    # to the CLI of cli/train.py (PyTorch) or cli/train_deepar_gluonts.py (DeepAR).
    base_params = {
        "model": model,
        "csv_path": f"/opt/ml/input/data/training/{csv_file}",
        "pred_len": pred_len,
        "epochs": epochs,
        "seed": 42,
        "workers": 2,
        "no_resume": True,     # disable resume for clean baselines (set False via set_hyperparameters below)
        "resume_key": "",
        "last_horizon_only": True,  # evaluate only the final horizon (DeepAR parity)
    }
    if enc_in is not None:
        base_params["enc_in"] = int(enc_in)

    # Merge dataset/model-specific knobs
    if model.lower() == "patchtst":
        seq_len, mcfg = cfg_patchtst(ds_key, pred_len, smd_granularity)
        base_params.update({"seq_len": seq_len, **mcfg})

    elif model.lower() == "itransformer":
        seq_len, mcfg = cfg_itransformer(ds_key, pred_len, smd_granularity)
        base_params.update({"seq_len": seq_len, **mcfg})

    else:
        # DeepAR uses a different entry point (GluonTS script); no src/ deps needed.
        entry_point = "train_deepar_gluonts.py"
        deps = []

        # Compute DeepAR-specific arguments (context length, freq, baseline HPs)
        ctx = _deepar_ctx_len(ds_key, pred_len, smd_granularity)
        freq = _freq_for(ds_key, smd_granularity)
        dph = _deepar_hp_for(ds_key)

        base_params = {
            "csv_path": f"/opt/ml/input/data/training/{csv_file}",
            "seq_len": ctx,             # DeepAR context_length
            "pred_len": pred_len,       # horizon
            "epochs": epochs,
            "batch_size": dph["batch_size"],
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "freq": freq,
            "seed": 42,
            "hidden_size": dph["hidden_size"],
            "num_layers": dph["num_layers"],
            "dropout": dph["dropout"],
        }
        # For DeepAR we also parse test MSE from stdout for convenience
        metric_defs = [
            {"Name": "val_mse", "Regex": r"val_mse=([0-9.]+)"},
            {"Name": "test_mse", "Regex": r"\[TEST\]\s+mse\s+([0-9.]+)"},
        ]

    # Build the Estimator object. This configures:
    # - entry point & source_dir (code upload)
    # - container image
    # - instance & spot policy
    # - S3 locations for outputs/checkpoints
    est = PyTorch(
        environment={"PYTHONHASHSEED": "0"},
        debugger_hook_config=False,
        disable_profiler=True,
        entry_point=entry_point,
        source_dir="cli",
        dependencies=deps,
        role=ROLE,
        framework_version="2.4",
        py_version="py311",
        instance_type=instance_type,
        instance_count=1,
        sagemaker_session=sess,
        code_location=code_loc,                   # where to upload code tarballs
        output_path=f"s3://{bucket}/outputs/",    # model artifacts & logs
        metric_definitions=metric_defs,           # regex-extracted metrics
        hyperparameters=base_params,              # CLI args for the training script
        use_spot_instances=True,                  # enable EC2 Spot for cost savings
        image_uri=img,
        max_run=6 * 3600,                         # hard limit (seconds) for training container
        max_wait=12 * 3600,                       # max wall-clock including spot queueing
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints/",  # rolling/spot resume state
    )

    # Helpful prints for debugging in CI/console
    print("[ENTRY_POINT]", entry_point)
    print("[IMAGE]", img)
    print("[INSTANCE]", instance_type)
    print("[HYPERPARAMETERS]", base_params)
    return est

# ----------------------------
# Workflow
# ----------------------------
def run_baseline(args):
    """
    Baseline workflow: iterate over horizons and submit one training job per horizon.
    - Builds the estimator
    - Sets run_name and resume_key (for spot/rolling resume)
    - Launches training and streams logs
    """
    ds_tag = Path(args.dataset).stem.upper()
    for H in args.horizons:
        # Build estimator for each horizon H
        est = build_estimator(
            args.model, args.instance, args.dataset, H, args.epochs,
            enc_in=args.enc_in, smd_granularity=args.smd_granularity
        )
        # Unique, readable, and SM-safe job name
        job_name = _mk_job_name(args.model, H, args.dataset)

        # resume_key tags model/dataset/seq_len/horizon for spot restarts
        seq_len = est.hyperparameters().get("seq_len", 336)
        resume_key = f"{args.model}-{Path(args.dataset).stem}-L{seq_len}-H{H}"

        # Enable resume now that we have a deterministic key (per horizon)
        est.set_hyperparameters(run_name=job_name, no_resume=False, resume_key=resume_key)
        print("Estimator.instance_type =", est.instance_type)
        print("Estimator.image_uri     =", est.image_uri)

        # Submit the job. wait=True blocks this process until the job completes.
        # logs="All" streams CloudWatch logs to the local console.
        est.fit(inputs=inputs, job_name=job_name, wait=True, logs="All")
        print("Completed:", job_name)

# ----------------------------
# CLI entry point
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    print("[ARGS]", args)
    run_baseline(args)
