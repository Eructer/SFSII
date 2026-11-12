import torch
import torch.nn as nn

def iou(logits, y):
    """Function for getting the IoU score. Closer to 1 is better

    Parameters
    ----------
        logits ():
        y ():

    Returns
    -------
        float : IoU score

    """
    pred = logits.argmax(1)
    inter = ((pred == 1) & (y == 1)).sum().item()
    union = ((pred == 1) | (y == 1)).sum().item()
    return inter / max(union, 1)

def print_param_names(model):
    """Function for printing out model parameters

    Parameters
    ----------
        model (TinyUNet): Model

    """
    # Check
    for n, p in model.named_parameters():
        print(f"  {n:60s} {tuple(p.shape)} requires_grad={p.requires_grad}")

def lambda_params(model) -> dict:
    """Function for getting lambda params

    Parameters
    ----------
        model (TinyUNet): Model

    Returns
    -------
        dict : Model lambda params

    """
    return {n : p for n, p in model.named_parameters() if "lambda" in n.lower()}

def check_lambda0(sm_model, ibnn_model, x: torch.Tensor):
    """Function for checking models lambda value (IBNN specific)

    Parameters
    ----------
        sm_model (TinyUNet): Baseline model
        ibnn_model (TinyUNet): IBNN model to check
        x (torch.Tensor): Sample shape

    """

    # Check
    missing = ibnn_model.load_state_dict(sm_model.state_dict(), strict = False)
    print("load_state_dict:", missing)
    # Uhh
    sm_model.eval(), ibnn_model.eval()
    with torch.no_grad():
        diff = (sm_model(x) - ibnn_model(x)).abs().max().item()
    print(f"max |SM - IBNN| at lambda = 0 : {diff:.2e} (expect -0)")

def _starts(n, tile, step):
    s = list(range(0, max(n - tile, 0) + 1, step))
    if s[-1] + tile < n: s.append(n - tile)
    return s

@torch.no_grad()
def tiled_inference(model, img, tile=512, overlap=64, n_classes=2):
    _, H, W = img.shape                       # img: (1, H, W)
    out = torch.zeros(n_classes, H, W, device=img.device)
    cnt = torch.zeros(1, H, W, device=img.device)
    for y in _starts(H, tile, tile - overlap):
        for x in _starts(W, tile, tile - overlap):
            patch = img[:, y:y+tile, x:x+tile].unsqueeze(0)
            out[:, y:y+tile, x:x+tile] += model(patch)[0].float()
            cnt[:, y:y+tile, x:x+tile] += 1
    return out / cnt

@torch.no_grad()
def evaluate(model, loader, tile=512):
    model.eval(); inter = union = 0
    for x, y in loader:
        pred = tiled_inference(model, x[0].to(DEVICE), tile).argmax(0).cpu()
        t, p = y[0] == 1, pred == 1
        inter += (p & t).sum().item(); union += (p | t).sum().item()
    return inter / max(union, 1)


def soft_iou_loss(logits, y, eps=1e-6):
    p = logits.float().softmax(1)[:, 1]; t = (y == 1).float()
    inter = (p * t).sum()
    return 1 - (inter + eps) / (p.sum() + t.sum() - inter + eps)


def loss_fn(logits, y):
    return ce(logits.float(), y) + soft_iou_loss(logits, y)