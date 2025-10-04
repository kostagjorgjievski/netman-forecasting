import os, csv, random, numpy as np, torch

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def auto_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def csv_log(path: str, header: list, row: list):
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "a", newline = "") as f:
        w = csv.writer(f)
        if write_header: 
            w.writerow(header)
        w.writerow(row)