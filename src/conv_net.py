import torch
import torch.nn as nn
from modified_rf.nn_layers import SMLayer, IBNNLiteLayer, IBNNLayer


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
            w_kernel_size = [],
            in_size = (c_in, c_in)
            )

    elif kind == "ibnn":
        layer = IBNNLayer(
            **common,
            w_kernel_size = [],
            f_solver = "fixed_point_iter",
            f_max_iter = 5,
            f_tol = 1e-3,
            initial_lambda = 1e-4,
            in_size = (c_in, c_in)
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
        return self.bn(self.conv(x))

class UNet3(nn.Module):
    def __init__(self, in_ch=1, n_classes=2, kind="sm", ibnn_at=("mid",), width=16):
        super().__init__()
        k = lambda n: kind if n in ibnn_at else "sm"
        self.enc1 = Block(k("enc1"), in_ch, width)
        self.enc2 = Block(k("enc2"), width, 2 * width)
        self.mid  = Block(k("mid"), 2 * width, 4 * width)
        self.up2  = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.dec2 = Block(k("dec2"), 4 * width, 2 * width)
        self.up1  = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.dec1 = Block(k("dec1"), 2 * width, width)
        self.head = nn.Conv2d(width, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m  = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)