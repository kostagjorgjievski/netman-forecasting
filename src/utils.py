import os, csv, random, numpy as np, torch

def set_seed(s: int):
    """
    Set RNG seeds for Python, NumPy, and PyTorch to improve reproducibility.

    Args:
        s: Seed value (integer).
    Notes:
        - Sets both CPU and CUDA seeds. For full determinism, you may also need to
          configure torch.backends.cudnn.deterministic / benchmark outside this helper.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def auto_device():
    """
    Pick a reasonable default torch.device:
      - Prefer CUDA if available,
      - else prefer Apple Metal (MPS) if available,
      - otherwise fall back to CPU.

    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: torch.nn.Module

    Returns:
        int: total number of parameters with requires_grad=True
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def csv_log(path: str, header: list, row: list):
    """
    Append a single row to a CSV file, writing the header if the file doesn't exist.

    Args:
        path:   Output CSV path.
        header: List of column names (written only once when file is created).
        row:    List of values for a single record/line.

    Behavior:
        - Creates parent directories if missing.
        - Opens the file in append mode and writes one row.
        - Uses newline='' to avoid extra blank lines on Windows.
    """
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
