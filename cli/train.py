# cli/train.py
import argparse, os, math, torch
from datetime import datetime
from src.models import build_model
from src.data.datasets import build_dataloaders
from src.train_eval import train_epoch, evaluate
from src.utils import set_seed, auto_device, count_params, csv_log
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()

    # core
    ap.add_argument("--model", type=str, required=True,
                    choices=["itransformer", "patchtst", "deepar"])
    ap.add_argument("--csv_path", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96,
                    choices=[96, 192, 336, 720])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)

    # model hyperparameters (unused ones can be ignored by specific models)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--activation", type=str, default="gelu")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--embed", type=str, default="timeF")
    ap.add_argument("--freq", type=str, default="h")
    ap.add_argument("--use_norm", action="store_true")
    ap.add_argument("--output_attention", action="store_true")

    # optimization
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)

    # I/O
    ap.add_argument("--logdir", type=str, default="results")
    ap.add_argument("--run_name", type=str, default=None)

    # PatchTST specific:
    ap.add_argument("--patch_len", type=int, default=16)
    ap.add_argument("--stride",    type=int, default=16)
    ap.add_argument("--revin",     action="store_true")

    

    # SageMaker-friendly paths (work locally too)
    ap.add_argument("--data_dir", type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", os.getcwd()))
    ap.add_argument("--checkpoint_dir", type=str,
        default=os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    ap.add_argument("--no_resume", action="store_true",
                help="Ignore rolling checkpoint (last.pt) and start fresh for this run.")


    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = auto_device()

    # lightweight config object that model constructors expect
    class Cfg: pass
    cfg = Cfg()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    if not hasattr(cfg, "class_strategy"):
        cfg.class_strategy = "projection"

    # --- PatchTST config shim: infer enc_in (D) and fill defaults the repo expects ---
    if args.model.lower() == "patchtst":
        def _infer_enc_in(path: str) -> int:
            # try with header, then without (handles both)
            for skip in (1, 0):
                try:
                    X = np.genfromtxt(path, delimiter=",", skip_header=skip)
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    if X.size > 0:
                        return X.shape[1]
                except Exception:
                    pass
            raise RuntimeError(f"Could not infer enc_in (num columns) from {path}")

        cfg.enc_in = _infer_enc_in(args.csv_path)   # <- key field PatchTST needs

        # Fill other attributes that the official wrapper accesses directly
        defaults = dict(
            fc_dropout=args.dropout,
            head_dropout=args.dropout,
            individual=False,
            patch_len=getattr(args, "patch_len", 16),
            stride=getattr(args, "stride", 16),
            padding_patch=None,
            revin=getattr(args, "revin", False),
            affine=False,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
            max_seq_len=1024,
            d_k=None, d_v=None,
            norm="BatchNorm",
            attn_dropout=0.0,
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


    model = build_model(args.model, cfg).to(device)
    params = count_params(model)
    print(f"[{args.model}] device={device} params={params/1e6:.2f}M")


    # dataloaders (CSV path is explicit)
    dl_tr, dl_va, dl_te = build_dataloaders(
        csv_path=args.csv_path,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch=args.batch_size,
        workers=args.workers
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # paths
    ckpt_root = os.path.join(args.logdir, "ckpts")
    os.makedirs(ckpt_root, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)  # for rolling trainer state

    dataset_name = os.path.basename(args.csv_path).split(".")[0]
    run_name = args.run_name or f"{args.model}_{dataset_name}_L{args.seq_len}_H{args.pred_len}_{datetime.now().strftime('%m%d-%H%M')}"
    best_model_path = os.path.join(ckpt_root, f"{run_name}.pth")
    rolling_path = os.path.join(args.checkpoint_dir, "last.pt")  # for Spot resume

    # Spot-friendly resume (model + optimizer + epoch)
    start_epoch = 1
    if (not args.no_resume) and os.path.exists(rolling_path):
        try:
            state = torch.load(rolling_path, map_location=device)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optim"])
            start_epoch = int(state.get("epoch", 0)) + 1
            print(f"Resumed training from {rolling_path} at epoch {start_epoch}.")
        except Exception as e:
            print(f"Found rolling checkpoint but failed to load ({e}); Training from scratch.")

    best = math.inf
    for ep in range(start_epoch, args.epochs + 1):
        tr = train_epoch(model, dl_tr, opt, device)
        vmse, vmae = evaluate(model, dl_va, device)
        print(f"ep {ep:02d} | train {tr:.4f} | val mse {vmse:.4f} | val mae {vmae:.4f}")

        # track best model by validation MSE
        if vmse < best:
            best = vmse
            torch.save(model.state_dict(), best_model_path)
        
        # save rolling trainer state every epoch so interruptions can resume
        torch.save({
            "model": model.state_dict(), 
            "optim": opt.state_dict(), 
            "epoch": ep + 1
        }, rolling_path)


    # test with best weights (fallback to rolling checkpoint if best not saved yet)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif os.path.exists(rolling_path):
        print("No best .pth yet; using latest rolling checkpoint for test.")
        state = torch.load(rolling_path, map_location=device)
        model.load_state_dict(state["model"])
    else:
        print("No checkpoints found; testing current in-memory weights.")
    tmse, tmae = evaluate(model, dl_te, device)

    print(f"[TEST] mse {tmse:.4f} | mae {tmae:.4f}")

    # single-row CSV log
    header = ["model", "dataset", "seq_len", "pred_len", "seed", "mse", "mae", "lr", "batch", "epochs", "params"]
    row = [args.model, os.path.basename(args.csv_path), args.seq_len, args.pred_len,
           args.seed, f"{tmse:.6f}", f"{tmae:.6f}", args.lr, args.batch_size, args.epochs, params]
    csv_log(os.path.join(args.logdir, f"{args.model}.csv"), header, row)

if __name__ == "__main__":
    main()
