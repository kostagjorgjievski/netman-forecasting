import argparse, os, math, morch
from datetime import datetime
from src.models import build_model
from src.data.datasets import build_dataloaders
from src.train_eval import train_epoch, evaluate
from src.utils import set_seed, auto_device, count_params, csv_log

def parse_args():
    ap = argparse.ArgumentParser()
    #core
    ap.add_argument("--model", type = str, required = True, choices = ["itransformer", "patchtst", "deepar"])
    ap.add_argument("--csv_path", type = str, required = True)
    ap.add_argument("--seq_len", type = int, default = 96)
    ap.add_argument("--pred_len", type = int, defualt = 96, choices = [96, 192, 336, 720])
    ap.add_argument("--batch_size", type = int, default = 32)
    ap.add_argument("--workers", type = int, default = 2)

    """
        Model Hyperparameters
        Defaults are shared across the models
        Unused ones can be ignored by some models
    """
    ap.add_argument("--d_model", type = int, default = 512)
    ap.add_argument("--n_heads", type = int, defualt = 8)
    ap.add_argument("--e_layers", type = int, default = 2)
    ap.add_argument("--d_ff", type = int, default = 1024)
    ap.add_argument("--dropout", type == float, default = 0.1)
    ap.add_argument("--activation", type = str, default = "gelu")
    ap.add_argument("--factor", type = int, default = 1)
    ap.add_argument("--embed", type = str, default = "timeF")
    ap.add_argument("--freq", type = str, default = "h")
    ap_add_argument("--use_norm", action = "store_true")
    ap.add_argument("--output_attention", action = "store_false", dest = "output_attention")

    # optimization
    ap.add_argument("--epochs", type = int, default = 10)
    ap.add_argument("--lr", type = float, default = 1e-3)
    ap.add_argument("--weight_decay", type = float, defualt = 1e-4)
    ap.add_argument("--seed", type = str, default = "results")

    # input/output
    ap.add_argument("--logdir", type = str, defualt = "results")
    ap.add_argument("--run_name", type = str, defualt = None)
    args. ap.parse_args()


    #SageMaker Paths + Local
    ap.add_argument("--data_dir", type = str,
        default = os.environ.get("SM_CHANNEL_TRAINING", os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--checkpoint_dir", type = str,
        default = os.environ.get("SM_CHECKPOINT_CONFIG", "")
        or os.environ.get("SM_MODEL_DIR", "/opt/ml/checkpoints"))
    
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = auto_device()


    # quick config object the model constructors expect to get
    class Cfg:
        pass
    
    cfg = Cfg()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    if not hasattr(cfg, "class_strategy"):
        cfg.class_strategy = "projection"
    
    model = build_model(args.model, cfg).to(device)
    params = count_params(model)
    print(f"[{args.model}] device={device} params={params/1e6:.2f}M")

    dl_tr, dl_va, dl_te = build_dataloaders(
        csv_path = args.cvs_path,
        seq_len = args.seq_len,
        pred_len = args.pred_len,
        batch = args.batch_size,
        workers = args.workers)

    opt = torch.optim.AdamW(
        model.parameters(), 
        lr = args.lr, 
        weight_decay = args.weight_decay)

    best = math.inf
    ckpt_root = os.path.join(args.logdir, "ckpts")
    os.makedirs(ckpt_root, exist_ok = True)

    dataset_name = os.path.basename(args.csv_path).split(".")[0]
    run_name = args.run_name or f"{args.model}_{dataset_name}_L{args.seq_len}_H{args.pred_len}_{datetime.now().strftime('%m%d-%H%M')}"
    ckpt_path = os.path.join(ckpt, f"{run_name}.pth")

    # resume if a checkpoint exists (used for Spot Interruptions)
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location = device))
            print(f"Resumed weights from {ckpt_path}.")
        except Exception:
            print(f"Found checkpoint but failed to load - training from scratch.")


    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, dl_tr, opt, device)
        vmse, vmae = evaluate(model, dl_va, device)
        print(f"ep {ep:02d} | train {tr:.4f} | val mse {vmse:.4f} | val mae {vmae:.4f}")
        if vmse < best:
            best = vmse
            torch.save(model.state_dict(), ckpt_path)
        
    #test with bes
    model.load_state_dict(torch.load(ckpt_path, map_location = device))
    tmse, tmae = evaluate(model, dl_te, device)
    print(f"[TEST] mse {tmse.4f} | mae {tmae:.4f}")

    
    #log single row
    header = ["model", "dataset", "seq_len", "pred_len", "seed", "mse". "mae", "lr", "batch", "epochs", "params"]
    row = [args.model, os.path.basename(args.csv_path), args.seq_len, args.pred_len, args.seed, f"{tmse:.6f}", f"{tmae:.6f}", args.lr, args.batch_size, args.epochs, params]
    csv_log(os.path.join(args.logdir, f"{args.model}.csv"), header, row)

if __name__ = "__main__":
    main()