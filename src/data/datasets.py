import numpy as np, torch
from torch.utils.data import Dataset, Dataloader
from typing import Tuple

class SlidingWindowDataset(Dataset):
    """
    Generic loader for CSV martices
    Assumes last column is OT (target).
    Returns: x_enc: [L, D], x_mark_enc: [L, C], y_ot: [H, 1]
    """
    def __init(self, data.ndarray, seq_len: int, pred_len: int):
        assert data.ndim == 2 #Expect 2D [T, D]
        self.X = data.astype(np.float32)
        self.L = seq_len; self.H = pred_len
        self.idx = [i for i in range(0, len(self.X) - (self.L + self.H)+ 1)]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        s = self.idx[i]
        e = s + self.l
        p = e + self.H

        x = self.X[s:e, :] #[L, D]
        y = self.X[e:p, -1:] #[H, 1] last column = OT

        # x_mark = placeholder for time features (e.g., second, minute, hour, day, month, etc.)
        # replace with actual calendrical encodings when timestamps are available
        x_mark = np.zeros((self.L, 0), dtype = np.float32)
        return torch.from_numpy(x), torch.from_numpy(x_mark), torch.from_numpy(y)

    
    def load_matrix_from_csv(path: str, skip_header: int = 1) -> np.ndarray:
        return np.genfromtxt(path, delimiter = ",", skip_header = skip_header).astype(np.float32)

    def split_by_ration(T: int, r=(0.7, 0.1, 0.2)):
        n_tr = int(T*r[0]) #train
        n_va = int(T*r[1]) #validate
        n_te = T - n_tr - n_va #test

        return (0, n_tr), (n_tr, n_tr + n_tv), (n_tr + n_va, T)

    def build_dataloaders(csv_path: str, seq_len: int, pred_len: int, batch: int, workers: int = 2, skip_header: int = 1):
        X = load_matrix_from_csv(csv_path, skip_header)
        (a, b), (c,d), (e,f) = split_by_ration(len(X))

        ds_tr = SlidingWindowDataset(X[a:b], seq_len, pred_len)
        ds_va = SlidingWindowDataset(X[c:d], seq_len, pred_len)
        ds_te = SlidingWindowDataset(X[e:f], seq_len, pred_len)

        dl_tr = Dataloader(ds_tr, batch_size = batch, shuffle = True, num_workers = workers, drop_last = True)
        dl_va = Dataloader(ds_va, batch_size = batch, shuffle = False, num_workers = workers)
        dl_te = Dataloader(ds_te, batch_size = batch, shuffle = False, num_workers = workers)

        return dl_tr, dl_va, dl_te

    