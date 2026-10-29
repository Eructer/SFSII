"""Convert MNIST IDX (ubyte) files into PNG images (and optional segmentation masks).

Example:
    python mnist_ubyte_to_png.py \
        --images train-images-idx3-ubyte --labels train-labels-idx1-ubyte \
        --out data/mnist_png --split train --masks --mask-mode binary

Output layout:
    out/split/images/00000_label5.png
    out/split/masks/00000_label5.png      (only with --masks)
"""
import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def _open(path):
    path = str(path)
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def read_idx_images(path):
    with _open(path) as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path}: bad image magic number {magic} (expected 2051)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_idx_labels(path):
    with _open(path) as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path}: bad label magic number {magic} (expected 2049)")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if len(data) != n:
        raise ValueError(f"{path}: expected {n} labels, got {len(data)}")
    return data


def make_mask(img, label, threshold, mode):
    """MNIST has no ground-truth masks, so we derive them from pixel intensity.

    binary: 0 = background, 255 = digit  (view-friendly; divide by 255 in the loader)
    class : 0 = background, label+1 = digit  (values 1..10, for multi-class segmentation)
    """
    fg = img > threshold
    if mode == "binary":
        return (fg * 255).astype(np.uint8)
    return (fg * (int(label) + 1)).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="*-images-idx3-ubyte(.gz)")
    p.add_argument("--labels", required=True, help="*-labels-idx1-ubyte(.gz)")
    p.add_argument("--out", required=True, help="output root directory")
    p.add_argument("--split", default="train", help="subfolder name, e.g. train/test")
    p.add_argument("--masks", action="store_true", help="also write segmentation masks")
    p.add_argument("--mask-mode", choices=["binary", "class"], default="binary")
    p.add_argument("--threshold", type=int, default=0,
                   help="pixel > threshold counts as digit (default 0; try 30-127 to drop faint edges)")
    args = p.parse_args()

    images = read_idx_images(args.images)
    labels = read_idx_labels(args.labels)
    assert len(images) == len(labels), "image/label count mismatch"

    root = Path(args.out) / args.split
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    if args.masks:
        mask_dir = root / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, lab) in enumerate(zip(images, labels)):
        name = f"{i:05d}_label{lab}.png"
        Image.fromarray(img, mode="L").save(img_dir / name)
        if args.masks:
            m = make_mask(img, lab, args.threshold, args.mask_mode)
            Image.fromarray(m, mode="L").save(mask_dir / name)

    print(f"Wrote {len(images)} images to {img_dir}" + (f" and masks to {mask_dir}" if args.masks else ""))


if __name__ == "__main__":
    main()