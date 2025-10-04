import torch, torch.nn as nn

class Averager:
    def __init__(self):
        self.n = 0
        self.s = 0.0
    
    def add(self, v, k=1):
        self.s += float(v) * k
        self.n += k
    
    def avg(self):
        return self.s / max(self.n, 1)

def _unpack(batch):
    # Accept (x, xmark, y) or (x, xmark, y, ...) and ignore any extras
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        x, xmark, y = batch[0], batch[1], batch[2]
        return x, xmark, y
    raise ValueError(f"Unexpected batch structure: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'n/a'}")

def train_epoch(model, loader, optim, device, amp=True):
    model.train()
    mse = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == "cuda"))
    loss_avg = Averager()

    for batch in loader:
        x, xmark, y = _unpack(batch)
        x, xmark, y = x.to(device), xmark.to(device), y.to(device)

        x_dec = torch.empty(0, device = device)
        xmark_dec = torch.empty(0, device = device)

        optim.zero_grad(set_to_none = True)
        with torch.autocast(device_type = device.type, dtype = torch.float16, enabled = (amp  and device.type == "cuda")):
            out = model(x, xmark, x_dec, xmark_dec) # [B, H, D]
            if isinstance(out, (list, tuple)): # handle output_attention=True
                out = out[0]
            out_ot = out[..., -1:] # OT = last channel
            loss = mse(out_ot, y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        loss_avg.add(loss.item(), x.size(0))
    return loss_avg.avg()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse = nn.MSELoss(reduction = "mean")
    mae = nn.L1Loss(reduction = "mean")
    m, a = Averager(), Averager()


    for batch in loader:
        x, xmark, y = _unpack(batch)
        x, xmark, y = x.to(device), xmark.to(device), y.to(device)
        out = model(x, xmark, torch.empty(0, device = device), torch.empty(0, device = device))
        if isinstance(out, (list, tuple)): # handle output_attention=True
            out = out[0]
        out_ot = out[..., -1:]
        m.add(mse(out_ot, y).item(), x.size(0))
        a.add(mae(out_ot, y).item(), x.size(0))
    return m.avg(), a.avg()