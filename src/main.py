import time
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
from scipy import ndimage
from itertools import combinations
import math
# Local imports
from conv_net import UNet3
from data import FilamentDataset
from util import (
    iou,
    print_param_names,
    lambda_params,
    check_lambda0,
    tiled_inference,
    evaluate_pq,
    soft_iou_loss,
    loss_fn,
    new_stats,
    semantic_to_panoptic,
    pq_image,
    pq_report,
    save_checkpoint,
    load_checkpoint,
    threshold_sweep,
    to_instances,
    decide
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



if DEVICE == "cpu":
    print("Using cpu")
    torch.set_num_threads(10)
    torch.set_num_interop_threads(2)
else:
    print("Using GPU")

# Train and Eval section below ↓
from itertools import islice

def run_validation(model, val_data, best, config, ep, max_images=None,
                   per_class=False, tag=""):
    """Runs validation, saves best.pt if improved. Returns the new best score."""
    model.eval()
    limited = lambda: islice(val_data, max_images) if max_images else val_data

    agn = evaluate_pq(model, limited(), n_classes=5, tile=512,
                      class_agnostic=True, DEVICE=DEVICE)
    score = agn["mean_PQ"]
    print(f"\n[{tag}] class-agnostic PQ: {score:.3f}")

    if per_class:                                   # second inference pass, so it's slower
        rep = evaluate_pq(model, limited(), n_classes=5, tile=512, DEVICE=DEVICE)
        print({c: round(v["PQ"], 3) for c, v in rep.items() if c != "mean_PQ"},
              "mean", round(rep["mean_PQ"], 3))

    if score > best:
        best = score
        torch.save({"model": model.state_dict(), "config": config,
                    "epoch": ep, "score": float(score)}, "src/models/best.pt")
        print(f"  new best ({score:.3f}) saved")

    model.train()
    torch.cuda.empty_cache()
    return best

def train(model, train_data, val_data, epochs, ce, config=None,
          ckpt_every=200, resume=True, seed=0, lr=1e-3, val_every_epochs = 2, val_every_step = None, val_max_images=20, val_per_class=False):
    """Same as before, plus checkpointing. config = dict describing the model
    (arch, kind, ibnn_at, width, n_classes, tile) so it can be rebuilt later."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    latest = Path("src/models/latest.pt")
    best_path = Path("src/models/best.pt")

    start_ep, start_step, best, history = 0, 0, -1.0, []
    if resume and latest.exists():
        try:
            start_ep, start_step, best, history = load_checkpoint(latest, model, opt, DEVICE)
        except Exception as e:                          # corrupted? fall back to the older copy
            print("latest.pt failed to load:", e)
            start_ep, start_step, best, history = load_checkpoint("src/models/latest_prev.pt", model, opt, DEVICE)
        print(f"Resuming at epoch {start_ep}, step {start_step}, best {best:.4f}")

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\n-----Training-----\n")
    t0 = time.time()

    for ep in range(start_ep, epochs):
        print(f"Epoch : {ep}", end='\n')
        model.train()

        # Same shuffle order for a given epoch, so a resume sees the same sequence
        train_data.sampler.generator = torch.Generator().manual_seed(seed + ep)

        train_ind = start_step if ep == start_ep else 0
        loader_iter = iter(train_data)
        if train_ind:                                   # skip batches already trained on
            loader_iter = islice(loader_iter, train_ind, None)

        for x, y in loader_iter:
            start = time.time()

            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(x), y, ce)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_ind += 1
            history.append(loss.item())
            end = time.time()

            print(f"\tTrain : {train_ind} | Time : {end-start:.3f} secs | "
                  f"loss(avg50) {np.mean(history[-50:]):.4f} | "
                  f"Alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB", end='\r')

            if train_ind % ckpt_every == 0:
                save_checkpoint(latest, model, opt, ep, train_ind, best, config, history)

        # ---- end of epoch ----
        is_last = (ep == epochs - 1)
        if is_last or (val_every_epochs and (ep + 1) % val_every_epochs == 0):
            best = run_validation(model, val_data, best, config, ep,
                                  None if is_last else val_max_images,    # full val set at the very end
                                  val_per_class or is_last, tag=f"epoch {ep}")

        # Epoch finished: record that the next epoch starts at step 0
        save_checkpoint(latest, model, opt, ep + 1, 0, best, config, history)

        Path("src/models").mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config, "epoch": ep},
                   f"src/models/epoch_{ep}.pt")         # optional: one file per epoch

        torch.cuda.empty_cache()

    secs = time.time() - t0
    tstr = f"Train time (min) : {secs//60}" if secs > 60 else f"Train time (min): {secs}"
    print(tstr)

    for n, p in lambda_params(model).items():
        print(f"    {n} : mean = {p.mean().item():.4f}")

    return model

CLASS_NAMES = ["background", "Left", "Right", "Unidentifiable", "Ambiguous"]

@torch.no_grad()
def test(model, test_loader, n_classes=5, tile=512, overlap=64,
              n_vis=3, out_dir="results", use_amp=False):
    model.eval()
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)

    conf = np.zeros((n_classes, n_classes), dtype=np.int64)   # rows = truth, cols = prediction
    st_cls = new_stats(n_classes)      # PQ with classes
    st_agn = new_stats(n_classes)      # PQ ignoring handedness

    for i, (x, y, inst) in enumerate(test_loader):             # batch size 1, full images
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = tiled_inference(model, x[0].to(DEVICE), tile, overlap, n_classes)
        probs = logits.float().softmax(0).cpu().numpy()
        pred = probs.argmax(0)
        gt = y[0].numpy()

        # pixel-level confusion matrix
        conf += np.bincount((gt * n_classes + pred).ravel(),
                            minlength=n_classes ** 2).reshape(n_classes, n_classes)

        # instance-level (PQ)
        pr_inst, pr_cls, _ = semantic_to_panoptic(probs)
        gt_inst = inst[0].numpy()
        ids, first = np.unique(gt_inst.ravel(), return_index=True)
        gt_cls = {int(k): int(gt.ravel()[f]) for k, f in zip(ids, first) if k > 0}

        pq_image(gt_inst, gt_cls, pr_inst, pr_cls, st_cls)
        pq_image(gt_inst, {k: 1 for k in gt_cls}, pr_inst, {k: 1 for k in pr_cls}, st_agn)

        if i < n_vis:                                          # save a few overlays
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
            ax[0].imshow(x[0, 0], cmap="gray"); ax[0].set_title("input")
            ax[1].imshow(gt, cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[1].set_title("truth")
            ax[2].imshow(pred, cmap="tab10", vmin=0, vmax=9, interpolation="nearest"); ax[2].set_title("prediction")
            for a in ax: a.axis("off")
            fig.savefig(out_dir / f"test_{i}.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    # pixel metrics from the confusion matrix
    tp = np.diag(conf).astype(float)
    iou_c = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)           # per class
    fg_inter = conf[1:, 1:].sum()
    fg_union = conf[1:, :].sum() + conf[:, 1:].sum() - fg_inter
    results = {
        "pixel_acc": tp.sum() / conf.sum(),
        "iou_per_class": dict(zip(CLASS_NAMES, iou_c.round(4))),
        "foreground_iou": fg_inter / max(fg_union, 1),                   # any filament vs background
        "pq_per_class": pq_report(st_cls),
        "pq_class_agnostic": pq_report(st_agn),
        "confusion": conf,
    }

    print(f"pixel acc        : {results['pixel_acc']:.4f}   (inflated by background)")
    print(f"foreground IoU   : {results['foreground_iou']:.4f}")
    for n, v in results["iou_per_class"].items():
        print(f"  IoU {n:15s}: {v:.4f}")
    print("PQ per class     :", {CLASS_NAMES[c]: round(v["PQ"], 3)
          for c, v in results["pq_per_class"].items() if c != "mean_PQ"},
          "| mean", round(results["pq_per_class"]["mean_PQ"], 3))
    print("PQ class-agnostic:", round(results["pq_class_agnostic"]["mean_PQ"], 3))
    return results, model

def save_model(model, epoch):
    """Function for saving model

    Parameters
    ----------
        model (TinyUNet): Model to save
    """
    torch.save(model.state_dict(), f"{epoch}_model.pt")

# Change sizes depending on image
PALETTE = np.array([
    (0, 0, 0),         # 0 background
    (230, 25, 75),     # 1 Left
    (60, 120, 255),    # 2 Right
    (255, 225, 25),    # 3 Unidentifiable
    (150, 150, 150),   # 4 Ambiguous
], dtype=np.uint8)

def segment_image(model, image_path, out_path="results/segmented.png", device=DEVICE,
                  n_classes=5, compare=False, tile=512, overlap=64,
                  fg_threshold=0.5, instances=False, min_area=50,
                  resize=None, use_amp=False, return_extra=False):
    """Segment one image with tiled inference.

    Parameters
    ----------
        model: trained network
        image_path (str): input image
        out_path (str): colored prediction image. Also writes <stem>_cls.png (raw class IDs,
                        values 0..n_classes-1) and, if compare, <stem>_compare.png
        n_classes (int): number of output channels of the model (5 = background + 4 classes)
        tile, overlap (int): tile size and overlap used for inference; use the training tile size
        fg_threshold (float): a pixel is filament if P(not background) > threshold.
                              Lower it (e.g. 0.2) if the model predicts too little
        instances (bool): label connected components and draw their bounding boxes in the plot
        min_area (int): drop components smaller than this (pixels)
        resize (int | None): resize the image to resize x resize first. None keeps the original size
        use_amp (bool): bf16 autocast, only if you trained that way
        return_extra (bool): also return the probability maps and the instance map

    Returns
    -------
        arr (np.ndarray): input image, (H, W) uint8
        pred (np.ndarray): class map, (H, W), values 0..n_classes-1
        probs, inst (only if return_extra): softmax probabilities (C, H, W), instance map (H, W)
    """
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    img = Image.open(image_path).convert("L")
    if resize:
        img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.array(img)
    x = torch.from_numpy(arr).float().div(255.0).unsqueeze(0).to(device)    # (1, H, W)

    # Tiled inference -> probabilities
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = tiled_inference(model, x, tile, overlap, n_classes)
    # probs = logits.float().softmax(0).cpu().numpy()                          # (C, H, W)


    logits = logits.float()
    lo_thr = math.log(fg_threshold / (1 - fg_threshold))      # 0.5 -> 0.0, 0.999 -> 6.9
    pred, lo = decide(logits, lo_thr)
    probs = logits.softmax(0).cpu().numpy() if return_extra else None

    inst = classes = boxes = None
    if instances:
        inst, classes, boxes = to_instances(pred, n_classes, min_area)

    # Class map: foreground where P(not background) > threshold, class = best foreground class
    # fg = 1.0 - probs[0]
    # if n_classes == 2:
    #     pred = (fg > fg_threshold).astype(np.uint8)
    # else:
    #     best = probs[1:].argmax(0) + 1
    #     pred = np.where(fg > fg_threshold, best, 0).astype(np.uint8)

    # Optional instances from connected components
    # inst = None
    # if instances:
    #     inst, _ = ndimage.label(pred > 0, structure=np.ones((3, 3)))
    #     for i, sl in enumerate(ndimage.find_objects(inst), start=1):
    #         if sl is not None and (inst[sl] == i).sum() < min_area:
    #             inst[sl][inst[sl] == i] = 0

    # Save raw class IDs and a colored image
    Image.fromarray(pred).save(out_path.with_name(out_path.stem + "_cls.png"))
    color = PALETTE[np.clip(pred, 0, len(PALETTE) - 1)]
    Image.fromarray(color).save(out_path)

    # Side-by-side comparison, saved to its own file
    if compare:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(arr, cmap="gray"); ax[0].set_title("Input")
        ax[1].imshow(color); ax[1].set_title("Prediction")
        ax[2].imshow(arr, cmap="gray")
        ax[2].imshow(np.ma.masked_equal(pred, 0), cmap="autumn", alpha=0.6, interpolation="nearest")
        ax[2].set_title("Overlay")
        if boxes:
            for i, (x0, y0, x1, y1) in boxes.items():
                ax[1].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                   fill=False, edgecolor="white", linewidth=1))
                if n_classes == 5:
                    ax[1].text(x0, y0, "LRUA"[classes[i] - 1], color="white", fontsize=6)
        for a in ax:
            a.axis("off")
        plt.show()
        fig.savefig(out_path.with_name(out_path.stem + "_compare.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"predicted classes: {dict(zip(*np.unique(pred, return_counts=True)))}")

    return (arr, pred, probs, inst) if return_extra else (arr, pred)

def from_scratch(
    epochs: int,
    bs: int = 32,
    save: bool = False,
    training_size: int = 50,
    val_size: int = 50,
    kind = "sm",
    crop_size=16,
    train_triples=list,
    val_triples=list,
    test_triples=list,
    train_data=FilamentDataset,
    val_data=FilamentDataset,
    test_data=FilamentDataset,
    train_loader=DataLoader,
    val_loader=DataLoader,
    test_loader=DataLoader
    ):
    """Function for getting the model from training

    Returns
    -------
        model (TinyUNet): Trained model

    """
    # Create structure of data for lambda check
    # x = torch.randn(16, 1, 2048, 2048, device = DEVICE)

    # Create small conv, no IBNN. Baseline kinda
    # sm = TinyUNet(kind = "sm").to(DEVICE)
    # Create lite IBNN
    model = UNet3(kind = kind, n_classes=5, ibnn_at=("mid","enc1")).to(DEVICE)

    # Full implicit layer
    # model = TinyUNet(kind = "full", ibnn_at=('enc1,)).to(DEVICE)

    # Print parameters
    print_param_names(model)
    # IBNN lambda check
    # check_lambda0(sm, model, x)



    counts = np.zeros(5)
    for _, cp, _ in train_triples[:100]:                  # a subset is enough
        counts += np.bincount(np.array(Image.open(cp)).ravel(), minlength=5)
    w = 1.0 / np.sqrt(counts / counts.sum() + 1e-8)
    w = w / w[1:].mean()                                  # scale so foreground classes average 1
    w[0] = w[0] * 0.5                                     # soften; background still dominates by count
    print(counts / counts.sum(), w)

    # If you use loss_fn from util
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))


    # Train model

    config = dict(arch="UNet3", kind=kind, ibnn_at=("mid",), width=16, n_classes=5, tile=512)
    print(config)
    model = train(model, train_loader, val_loader, epochs, ce, config=config)

    

    # Save model
    if save:
        print(f"-----Saving Model-----")
        save_model(model, f"{KIND}_{epochs}_{training_size}_{val_size}_{crop_size}")

    # Test model
    results, model = test(model, test_loader, n_classes=5, tile=512)

    return model, config, results

def load_model(path, device=DEVICE, cfg = None):
    ck = torch.load(path, map_location=device, weights_only=False)

    # Old-style files saved with torch.save(model.state_dict(), ...) are the weights themselves
    if "model" not in ck:
        raise ValueError("No 'model' key: this is a bare state_dict. "
                         "Build the model yourself, then call model.load_state_dict(ck).")

    cfg = ck.get("config") or {}
    arch = cfg.get("arch", "UNet3")
    cls = {"UNet3": UNet3, "TinyUNet": TinyUNet}[arch]

    model = cls(
        in_ch=1,
        n_classes=cfg.get("n_classes", 5),
        kind=cfg.get("kind", "sm"),
        ibnn_at=tuple(cfg.get("ibnn_at", ())),
        width=cfg.get("width", 16),
    ).to(device)

    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded {path} | epoch {ck.get('epoch')} | score {ck.get('score')} | config {cfg}")
    return model, cfg

def load_any(path, device=DEVICE):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cfg = ck.get("config") if isinstance(ck, dict) else None
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    tensor_shapes = lambda d: {k: tuple(v.shape) for k, v in d.items() if torch.is_tensor(v)}
    fs = tensor_shapes(sd)
    n_classes, width = sd["head.weight"].shape[:2]

    # Architecture: from the saved config, otherwise from the key names
    if isinstance(cfg, dict) and cfg.get("arch") in ARCHS:
        arch = ARCHS[cfg["arch"]]
    else:
        arch = UNet3 
    blocks = ["enc1", "enc2", "mid", "dec2", "dec1"] if arch is UNet3 else ["enc1", "enc2", "mid", "dec1"]
    print("Architecture:", arch.__name__, "| width", int(width), "| n_classes", int(n_classes))

    def finish(model, used):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad_m = [k for k in missing if "_extra_state" not in k]
        bad_u = [k for k in unexpected if "_extra_state" not in k]
        if bad_m or bad_u:
            raise RuntimeError(f"missing {bad_m[:5]}, unexpected {bad_u[:5]}")
        print("Loaded with", used)
        return model.to(device).eval(), (cfg or used)

    # 1. Saved config, if there is one
    if isinstance(cfg, dict) and "kind" in cfg:
        try:
            model = arch(in_ch=1, n_classes=int(n_classes), kind=cfg["kind"],
                         ibnn_at=tuple(cfg.get("ibnn_at", ())), width=int(width))
            return finish(model, cfg)
        except Exception as e:
            print("Saved config didn't load:", e)

    # 2. Otherwise search for the matching kind and ibnn_at
    candidates = [("sm", ())] + [(kind, s) for kind in ("lite", "ibnn")
                                 for r in range(1, len(blocks) + 1)
                                 for s in combinations(blocks, r)]
    best = None
    for kind, subset in candidates:
        try:
            model = arch(in_ch=1, n_classes=int(n_classes), kind=kind, ibnn_at=subset, width=int(width))
        except Exception:
            continue
        ms = tensor_shapes(model.state_dict())
        if ms == fs:
            return finish(model, dict(arch=arch.__name__, kind=kind, ibnn_at=subset,
                                      width=int(width), n_classes=int(n_classes)))
        diff = set(ms) ^ set(fs)
        if best is None or len(diff) < best[0]:
            best = (len(diff), kind, subset, diff)

    print("No exact match. Closest:", best[1], best[2], "| differing keys:", sorted(best[3])[:20])
    raise RuntimeError("Could not match this checkpoint to a model configuration")

if __name__ == "__main__":
    # Free memory
    # torch.cuda.empty_cache()



    epochs = 10
    training_size = 150
    val_size = 50
    crop_size = 8
    bs = 1
    KIND = "lite"
    config = dict(arch="UNet3", kind=KIND, ibnn_at=("mid","enc1"), width=16, n_classes=5, tile=512)
    # epochs = 10, training = 100, val = 50, crop = 8, kind = lite, ibnn at [mid], 30 hours, pretty bad
    # epochs = 10, training = 150, val = 50, crop = 8, kind = lite, ibnn at [mid], 41.2 hours,
    # epochs = 10, training = 150, val = 50, crop = 8, kind = lite, ibnn at [mid, enc1], 41.2 hours,

    # Solar Filament
    manifest = json.load(open("data/mask_cls/manifest.json"))
    # Discover files once (tile=None, so nothing is cropped here)
    random.Random(45).shuffle(manifest)
    n = len(manifest)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    train_triples = manifest[:n_train]
    val_triples   = manifest[n_train:n_train + n_val]
    test_triples  = manifest[n_train + n_val:]

    train_data = FilamentDataset(train_triples[:training_size], tile=512, crops_per_image=crop_size, augment=True)
    val_data   = FilamentDataset(val_triples[:val_size], tile=None)
    test_data  = FilamentDataset(test_triples, tile=None)

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    val_loader   = DataLoader(val_data, batch_size=1, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=1, num_workers=2, pin_memory=True)

    # # Train and get model from scratch
    model, config, results = from_scratch(
        epochs,
        kind=KIND,
        save=True,
        bs=bs,
        training_size=training_size,
        val_size=val_size,
        crop_size=crop_size,
        train_triples=train_triples,
        val_triples=val_triples,
        test_triples=test_triples,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
        )

    # model, cfg = load_any(f"../{KIND}_{epochs}_{training_size}_{val_size}_{crop_size}_model.pt")

    # Test different thresholds for optimal pick
    #threshold_sweep(model, val_loader, DEVICE)

    # torch.cuda.empty_cache()
    target_files = [
        "data/filament/test/test_images/20210730044930Lh.jpeg",
        "data/filament/test/test_images/20211119005210Mh.jpeg",
        "data/filament/test/test_images/20211214174010Mh.jpeg"
    ]
    ind = 0
    for file in target_files:
        arr, pred = segment_image(
            model,
            file,
            out_path=f"results/{ind}_{epochs}_{training_size}_{val_size}.png",
            compare=True, 
            device=DEVICE,
            n_classes=5,
            tile=512, 
            fg_threshold=0.989, # Model has to be confident at % for predicting
            instances=True)
        ind += 1

    # Further improvments, add ibnn at mid and enc1, more images, more epochs