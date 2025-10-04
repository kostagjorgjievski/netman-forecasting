import numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple

class SlidingWindowDataset(Dataset):
    """
    Generic loader for CSV martices (T, D). Assumes last column is OT (target).
    Returns:
        x       : [L, D]
        x_mark  : [L, C] (placeholder zeros; time features depend on dataset)
        x_ot    : [H, 1]
    """

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        assert data.ndim == 2 #Expect 2D [T, D]
        self.X = data.astype(np.float32)
        self.L = seq_len; self.H = pred_len
        self.idx = [i for i in range(0, len(self.X) - (self.L + self.H)+ 1)]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s = self.idx[i]
        e = s + self.L
        p = e + self.H

        x = self.X[s:e, :] #[L, D]
        y = self.X[e:p, -1:] #[H, 1] last column = OT

        # x_mark = placeholder for time features (e.g., second, minute, hour, day, month, etc.)
        # replace with actual calendrical encodings when timestamps are available
        x_mark = np.zeros((self.L, 0), dtype = np.float32)
        return torch.from_numpy(x), torch.from_numpy(x_mark), torch.from_numpy(y)

    @staticmethod
    def load_matrix_from_csv(path: str, skip_header: int = 1) -> np.ndarray:
        arr = np.genfromtxt(path, delimiter = ",", skip_header = skip_header).astype(np.float32)
        if arr.ndim == 1: # handle single-column CSVs
            arr = arr.reshape(-1, 1)
        return arr

    @staticmethod
    def split_by_ratio(T: int, r=(0.7, 0.1, 0.2)):
        n_tr = int(T*r[0]) #train
        n_va = int(T*r[1]) #validate
        n_te = T - n_tr - n_va #test

        return (0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, T)

    @staticmethod
    def build_dataloaders(csv_path: str,
    seq_len: int, 
    pred_len: int, 
    batch: int, 
    workers: int = 2, 
    skip_header: int = 1, 
    pin_memory: bool = False):

        X = SlidingWindowDataset.load_matrix_from_csv(csv_path, skip_header)

        ds_full = SlidingWindowDataset(X, seq_len, pred_len)

        N = len(ds_full)
        n_tr = int(N * 0.7)
        n_va = int(N * 0.1)
        n_te = N - n_tr - n_va
        idx = list(range(N))

        ds_tr = Subset(ds_full, idx[:n_tr])
        ds_va = Subset(ds_full, idx[n_tr:n_tr+n_va])
        ds_te = Subset(ds_full, idx[n_tr+n_va:])

        dl_tr = DataLoader(ds_tr, batch_size = batch, shuffle = True, num_workers = workers, drop_last = True, pin_memory = pin_memory)
        dl_va = DataLoader(ds_va, batch_size = batch, shuffle = False, num_workers = workers, pin_memory = pin_memory)
        dl_te = DataLoader(ds_te, batch_size = batch, shuffle = False, num_workers = workers, pin_memory = pin_memory)

        return dl_tr, dl_va, dl_te


build_dataloaders = SlidingWindowDataset.build_dataloaders