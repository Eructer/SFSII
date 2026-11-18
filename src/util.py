import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import os
from pathlib import Path

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
def evaluate_pq(model, loader, n_classes, tile=512, class_agnostic=False, DEVICE = "cpu"):
    model.eval()
    stats = new_stats(n_classes)
    for x, y, inst in loader:                           # full images, batch size 1
        logits = tiled_inference(model, x[0].to(DEVICE), tile, n_classes=n_classes)
        pr_inst, pr_cls, _ = semantic_to_panoptic(logits.softmax(0).cpu().numpy())
        gt_inst, cmap = inst[0].numpy(), y[0].numpy()
        ids, first = np.unique(gt_inst.ravel(), return_index=True)
        gt_cls = {int(i): int(cmap.ravel()[f]) for i, f in zip(ids, first) if i > 0}
        if class_agnostic:
            gt_cls = {i: 1 for i in gt_cls}; pr_cls = {i: 1 for i in pr_cls}
        pq_image(gt_inst, gt_cls, pr_inst, pr_cls, stats)
    return pq_report(stats)


def soft_iou_loss(logits, y, eps=1e-6):
    p = 1 - logits.float().softmax(1)[:, 0]           # P(any filament class)
    t = (y > 0).float()
    inter = (p * t).sum()
    return 1 - (inter + eps) / (p.sum() + t.sum() - inter + eps)

def loss_fn(logits, y, ce):
    return ce(logits.float(), y) + soft_iou_loss(logits, y)


def make_panoptic(info, anns_by_img):
    W, H = info["width"], info["height"]
    inst_img = Image.new("I", (W, H), 0)
    d = ImageDraw.Draw(inst_img)
    cls_of = {}
    anns = sorted(anns_by_img.get(info["id"], []), key=lambda a: -a["area"])  # big first, small on top
    for n, a in enumerate(anns, start=1):
        seg = a["segmentation"]
        polys = seg if isinstance(seg[0], list) else [seg]
        for p in polys:
            pts = list(zip(p[0::2], p[1::2]))
            if len(pts) >= 3:
                d.polygon(pts, fill=n)
        cls_of[n] = a.get("category_id", a.get("category"))
    inst = np.array(inst_img, dtype=np.int32)
    cls = np.zeros(inst.shape, np.uint8)
    for n, c in cls_of.items():
        cls[inst == n] = c
    return inst, cls



def new_stats(n_classes):
    return {k: np.zeros(n_classes) for k in ("tp", "fp", "fn", "iou")}

def pq_image(gt_inst, gt_cls, pr_inst, pr_cls, stats):
    g_ids, g_a = np.unique(gt_inst[gt_inst > 0], return_counts=True)
    p_ids, p_a = np.unique(pr_inst[pr_inst > 0], return_counts=True)
    g_area, p_area = dict(zip(g_ids.tolist(), g_a)), dict(zip(p_ids.tolist(), p_a))

    both = (gt_inst > 0) & (pr_inst > 0)
    M = int(pr_inst.max()) + 1
    keys, inter = np.unique(gt_inst[both].astype(np.int64) * M + pr_inst[both], return_counts=True)

    mg, mp = set(), set()
    for k, i in zip(keys.tolist(), inter.tolist()):
        g, p = divmod(k, M)
        if gt_cls[g] != pr_cls[p]:
            continue
        iou = i / (g_area[g] + p_area[p] - i)
        if iou > 0.5:                                  # >0.5 guarantees a unique match
            c = gt_cls[g]
            stats["tp"][c] += 1; stats["iou"][c] += iou
            mg.add(g); mp.add(p)
    for g in g_area:
        if g not in mg: stats["fn"][gt_cls[g]] += 1
    for p in p_area:
        if p not in mp: stats["fp"][pr_cls[p]] += 1

def pq_report(s):
    out = {}
    for c in range(1, len(s["tp"])):
        tp, fp, fn = s["tp"][c], s["fp"][c], s["fn"][c]
        if tp + fp + fn == 0:
            continue
        sq = s["iou"][c] / tp if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        out[c] = dict(PQ=sq * rq, SQ=sq, RQ=rq, TP=int(tp), FP=int(fp), FN=int(fn))
    out["mean_PQ"] = float(np.mean([v["PQ"] for v in out.values()]))
    return out


def decide(logits, lo_thr=0.0):
    """logits: torch (C, H, W). A pixel is filament if log-odds(any filament vs background) > lo_thr."""
    lo = torch.logsumexp(logits[1:], 0) - logits[0]
    cls = logits[1:].argmax(0) + 1
    pred = torch.where(lo > lo_thr, cls, torch.zeros_like(cls))
    return pred.cpu().numpy().astype(np.uint8), lo.cpu().numpy()

def to_instances(pred, n_classes, min_area=50):
    inst, _ = ndimage.label(pred > 0, structure=np.ones((3, 3)))
    classes, boxes = {}, {}
    for i, sl in enumerate(ndimage.find_objects(inst), start=1):
        if sl is None:
            continue
        sub = inst[sl] == i
        if sub.sum() < min_area:
            inst[sl][sub] = 0                              # drop tiny fragments
            continue
        votes = np.bincount(pred[sl][sub], minlength=n_classes)[1:]
        classes[i] = int(votes.argmax()) + 1               # majority class
        boxes[i] = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)   # x0, y0, x1, y1
    return inst.astype(np.int32), classes, boxes


def semantic_to_panoptic(probs, min_area = 50, lo_thr = 0.0):
    """probs: (C, H, W) numpy softmax. Returns instance map, {id: class}, {id: box}."""
    pred, _ = decide(logits.float(), lo_thr)
    return to_instances(pred, n_classes, min_area)

def save_checkpoint(path, model, opt, epoch, step, best, config, history):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,                  # epoch in progress
        "step": step,                    # steps already finished inside that epoch
        "best": float(best),
        "config": config,
        "history": history[-1000:],      # keep it small
    }, tmp)
    if path.exists():
        os.replace(path, path.with_name(path.stem + "_prev.pt"))
    os.replace(tmp, path)                # atomic: a power cut can't leave a half-written file

def load_checkpoint(path, model, opt, DEVICE):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    return ck["epoch"], ck["step"], ck["best"], ck.get("history", [])

@torch.no_grad()
def threshold_sweep(model, loader, DEVICE, n_images=10, tile=512,
                    thresholds=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98)):
    model.eval()
    fgs, gts = [], []
    for i, (x, y, _) in enumerate(loader):
        if i >= n_images: break
        logits = tiled_inference(model, x[0].to(DEVICE), tile, 64, 5)
        fgs.append((1 - logits.float().softmax(0)[0]).cpu().half())
        gts.append(y[0] > 0)
    true_frac = np.mean([g.float().mean().item() for g in gts])
    for t in thresholds:
        inter = union = 0; frac = 0.0
        for fg, gt in zip(fgs, gts):
            p = fg.float() > t
            inter += (p & gt).sum().item(); union += (p | gt).sum().item()
            frac += p.float().mean().item()
        print(f"thr {t:.2f} | IoU {inter/max(union,1):.4f} | predicted fg {100*frac/len(fgs):.2f}% (true {100*true_frac:.2f}%)")
    return fgs, gts