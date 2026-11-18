from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random
import torch

def flip_cls(m, axis):
    m = np.flip(m, axis).copy()                  # a mirror image reverses chirality
    a, b = m == 1, m == 2
    m[a], m[b] = 2, 1
    return m

class FilamentDataset(Dataset):
    def __init__(self, triples, tile=512, crops_per_image=16, fg_prob=0.7, augment=False):
        self.items = list(triples)
        self.tile = tile
        self.crops_per_image = crops_per_image if tile else 1
        self.fg_prob = fg_prob
        self.augment = augment

    def __len__(self):
        return len(self.items) * self.crops_per_image

    def __getitem__(self, index):
        ip, cp, np_ = self.items[index // self.crops_per_image]
        img = np.array(Image.open(ip).convert("L"))
        cls = np.array(Image.open(cp)).astype(np.int64)

        if not self.tile:                         # evaluation: full image + instance map
            inst = np.array(Image.open(np_)).astype(np.int64)
            x = torch.from_numpy(img).float().div(255).unsqueeze(0)
            return x, torch.from_numpy(cls), torch.from_numpy(inst)

        H, W = cls.shape; t = self.tile
        y = x0 = None
        if random.random() < self.fg_prob:
            present = [c for c in (1, 2, 3, 4) if (cls == c).any()]
            if present:
                c = random.choice(present)       # class-balanced center
                fg = np.flatnonzero(cls == c)
                cy, cx = divmod(int(fg[random.randrange(fg.size)]), W)
                y = min(max(cy - t // 2, 0), H - t)
                x0 = min(max(cx - t // 2, 0), W - t)
        if y is None:
            y, x0 = random.randint(0, H - t), random.randint(0, W - t)
        img = img[y:y + t, x0:x0 + t]; cls = cls[y:y + t, x0:x0 + t]

        if self.augment:
            if random.random() < 0.5: img, cls = np.flip(img, 1), flip_cls(cls, 1)
            if random.random() < 0.5: img, cls = np.flip(img, 0), flip_cls(cls, 0)
            k = random.randint(0, 3)             # 90° rotations preserve chirality
            img, cls = np.rot90(img, k), np.rot90(cls, k)

        x = torch.from_numpy(np.ascontiguousarray(img)).float().div(255).unsqueeze(0)
        return x, torch.from_numpy(np.ascontiguousarray(cls))

def example(train_triples):
    samples = random.Random(1).sample(train_triples, 6)
    fig, ax = plt.subplots(2, 6, figsize=(24, 8))
    for j, (ip, cp, _) in enumerate(samples):
        img = np.array(Image.open(ip).convert("L"))
        cls = np.array(Image.open(cp))
        print(Path(ip).name, "| image", img.shape, "mask", cls.shape,
            "| fg %.3f%%" % (100 * (cls > 0).mean()), "| classes", np.unique(cls).tolist())
        ys, xs = np.nonzero(cls)
        if len(ys) == 0:
            ax[0, j].set_title("no filaments"); continue
        m = 60
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, img.shape[0])
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, img.shape[1])
        ax[0, j].imshow(img[y0:y1, x0:x1], cmap="gray"); ax[0, j].set_title(Path(ip).name[:14])
        ax[1, j].imshow(img[y0:y1, x0:x1], cmap="gray")
        ax[1, j].imshow(np.ma.masked_equal(cls[y0:y1, x0:x1], 0), cmap="autumn", alpha=0.5, interpolation="nearest")
        for a in ax[:, j]: a.axis("off")
    plt.show()
