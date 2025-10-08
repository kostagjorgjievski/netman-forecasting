# cli/train_deepar_gluonts.py
import argparse, os, json, numpy as np
from datetime import datetime

from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
# from gluonts.torch.distributions import StudentTOutput  # optional (default is StudentT)
from gluonts.evaluation.backtest import make_evaluation_predictions

import warnings
warnings.filterwarnings(
    "ignore",
    message=r"Using a non-tuple sequence for multidimensional indexing",
    category=UserWarning,
    module=r"gluonts\.torch\.util"
)



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)      # context_length
    ap.add_argument("--pred_len", type=int, default=96)     # prediction_length
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--freq", type=str, default="h")        # use lowercase
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    # I/O
    ap.add_argument("--logdir", type=str, default="results")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.environ.get("SM_CHECKPOINT_DIR") or "./results/spot_ckpts",
    )
    return ap.parse_args()

def load_target_last_column(csv_path):
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
    return X[:, -1].astype(np.float32)  # OT = last column

def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # seeds
    np.random.seed(args.seed)
    try:
        import torch, random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    target = load_target_last_column(args.csv_path)
    T = len(target)
    L, H = args.seq_len, args.pred_len
    train_end = T - H
    if train_end <= L:
        raise ValueError(f"Not enough points (T={T}) for context {L} and horizon {H}.")

    train_series = target[:train_end]
    test_series  = target
    freq = args.freq.lower()
    start_ts = "2020-01-01 00:00:00"  # dummy anchor; freq defines spacing

    train_ds = ListDataset([{"start": start_ts, "target": train_series}], freq=freq)
    test_ds  = ListDataset([{"start": start_ts, "target": test_series }], freq=freq)

    est = DeepAREstimator(
        freq=freq,
        prediction_length=H,
        context_length=L,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_batches_per_epoch=50,  # adjust as you like
        # distr_output=StudentTOutput(),  # optional; default is StudentT
        trainer_kwargs={"max_epochs": args.epochs, "enable_progress_bar": False},
    )
    predictor = est.train(train_ds)

    forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor, num_samples=100)
    forecasts = list(forecast_it)
    series = list(ts_it)

    fc_mean = forecasts[0].mean  # [H]
    gt = series[0].values[-H:]
    mse = float(np.mean((fc_mean - gt) ** 2))
    mae = float(np.mean(np.abs(fc_mean - gt)))

    dataset_name = os.path.basename(args.csv_path).split(".")[0]
    run_name = args.run_name or f"deepar_official_{dataset_name}_L{L}_H{H}_{datetime.now().strftime('%m%d-%H%M')}"

    model_dir = os.environ.get("SM_MODEL_DIR", os.path.join(args.logdir, "model"))
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump({"mse": mse, "mae": mae, "prediction_length": H, "context_length": L}, f)

    out_csv = os.path.join(args.logdir, "deepar_official.csv")
    header = ["model", "dataset", "seq_len", "pred_len", "seed", "mse", "mae", "lr", "batch", "epochs", "params"]
    exists = os.path.exists(out_csv)
    with open(out_csv, "a") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        row = ["deepar_official", dataset_name, str(L), str(H), str(args.seed),
               f"{mse:.6f}", f"{mae:.6f}", str(args.lr), str(args.batch_size), str(args.epochs), "n/a"]
        f.write(",".join(row) + "\n")

    print(f"[TEST] mse {mse:.4f} | mae {mae:.4f}")

if __name__ == "__main__":
    main()
