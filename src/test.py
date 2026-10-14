import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from modified_rf.nn_layers import SMLayer, IBNNLiteLayer, IBNNLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ToyBlobs(Dataset):
    """
    Synthetic binary segmentation : Noisy image, mask = where a smooth blob is
    """

    def __init__(self, n = 256, size = 64, seed = 0):
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(n, 1, size, size, generator = g)
        smooth = F.avg_pool2d(noise, 9, stride = 1, padding = 4)
        self.masks = (smooth > 0.0).long().squeeze(1)
        self.images = 0.5 * self.masks.unsqueeze(1).float() + 0.5 * torch.randn(n, 1, size, size, generator = g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.masks[i]


def make_conv_like(kind: str, c_in: int, c_out: int, k: int = 3):
    """
    Conv-like layer factory
    
    Parameters
    ----------
        kind (str): The type you want
        c_in (int): Number of input channels
        c_out (int): Number of output channels
        k (int): Kernal size

    Returns
    -------
        layer (SMLayer | IBNNLiteLayer | IBNNLayer)
    """

    common = dict(
        in_channels = c_in,
        out_channels = c_out,
        phi_activation = "relu",
        m_kernel_size = k,
        m_padding = "same",
        m_initialization = "zeros",
    )

    if kind == "sm":
        layer = SMLayer(**common) #** -> unpacking a dict
    
    elif kind == "lite":
        layer = IBNNLiteLayer(
            **common,
            initial_lambda = 0.0,
            w_kernel_size = []
            )

    elif kind == "ibnn":
        layer = IBNNLayer(
            **common,
            w_kernel_size = [],
            f_solver = "fixed_point_iter",
            f_max_iter = 20,
            f_tol = 1e-4,
            )

    else:
        raise ValueError(kind)

    # Check
    layer.random_initialization()

    return layer

class Block(nn.Module):
    def __init__(self, kind: str, c_in: int, c_out: int):
        super().__init__()
        self.conv = make_conv_like(kind, c_in, c_out)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return self.b(self.conv(x))

class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 2, kind: str = "sm", ibnn_at: tuple = ("enc1"), width: int = 16):
        super().__init__()
        k = lambda name: kind if name in ibnn_at else "sm"
        self.enc1 = Block(k("enc1"), in_ch, width)
        self.enc2 = Block(k("enc2"), width, 2 * width)
        self.mid = Block(k("mid"), 2 * width, 2 * width)
        self.up = nn.ConvTranspose2d(2 * width, width, 2, stride = 2)
        self.dec1 = Block(k("dec1"), 2 * width, width)
        self.head = nn.Conv2d(width, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(e2)
        d1 = self.dec1(torch.cat([self.up(m), e1], dim = 1))
        return self.head(d1)

def iou(logits, y):
    pred = logits.argmax(1)
    inter = ((pred == 1) & (y == 1)).sum().item()
    union = ((pred == 1) | (y == 1)).sum().item()
    return inter / max(union, 1)

def print_param_names(model):
    # Check
    for n, p in mode.namde_parameters():
        print(f"  {n:60s} {tuple(p.shape)} requires_grad={p.requires_grad}")

def lambda_params(model):
    return {n : p for n, p in model.named_parameters() if "lambda" in n.lower()}

def check_lambda0(sm_model, ibnn_model, x):
    
    # Check
    missing = ibnn_model.load_state_dict(sm_model.state_dict(), strict = False)
    print("load_state_dict:", missing)
    # Uhh
    sm_model.eval(), ibnn_model.eval()
    with torch.no_grad():
        diff = (sm_model(x) - ibnn_model(x)).abs().max().item()
    print(f"max |SM - IBNN| at lambda = 0 : {diff:.2e} (expect -0)")

# Train and Eval section below ↓

def run(kind: str, ibnn_at, epochs: int = 5, seed: int = 0, bs: int = 16):
    torch.manual_speed(seed)
    train = DataLoader(ToyBlobs(256, seed = 0), batch_size = bs, shuffle = True)
    val = DataLoader(ToyBlobs(64, seed = 1), batch_size = bs)
    model = TinyUnet(kind = kind, ibnn_at = ibnn_at).to(DEVICE)
    # Check
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for x, y in train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Loss would need to be the correct segmentation one for comp
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eva()
        with torch.no_grad():
            scores = [iou(model(x.to(DEVICE)), y.to(DEVICE)) for x, y in val]
        print(f"[{kind:5s}] epoch {ep}  loss {loss.item():.3f}  val IoU {sum(scores)/len(scores):.3f}")
    secs = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else float("nan")
    for n, p in lambda_params(model).items():
        print(f"    {n} : mean = {p.mean().item():.4f}")
    return model

if __name__ == "__main__":
    x = torch.randn(2, 1, 64, 64, device = DEVICE)
    sm = TinyUNet(kind = "sm").to(DEVICE)
    lite = TinyUNet(kind = "lite", ibnn_at=("enc1",)).to(DEVICE)
    print_param_names(lite)
    check_lambda0(sm, lite, x)

    run("sm", ibnn_at = ())

    run("lite", ibnn_at = ("enc1",))

    # Full implicit layer
    # run("ibnn", ibnn_at=("enc1",))