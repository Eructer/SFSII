import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

json_path = "src/data/filament/train/MAGFiLO.json"
out = "src/data/fil_modified"

data = json.load(open(json_path))

info = data["images"][0]
img_root = Path("src/data/filament/train/train_images")
by_name = {f.name: f for f in img_root.rglob("*") if f.is_file()}

anns_by_img = defaultdict(list)
for a in data["annotations"]:
    anns_by_img[a["image_id"]].append(a)

def make_mask(info):
    W, H = info["width"], info["height"]
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for a in anns_by_img.get(info["id"], []):
        seg = a["segmentation"]
        polys = seg if isinstance(seg[0], list) else [seg]
        for p in polys:
            pts = list(zip(p[0::2], p[1::2]))
            if len(pts) >= 3:
                d.polygon(pts, fill=255)
    return np.array(m)

n_pos = 0
for info in data["images"]:
    m = make_mask(info)
    Image.fromarray(m).save(f"{out}/images/{info['id']}.png")
    n_pos += m.any()

print(len(data["images"]), "masks written,", n_pos, "contain filaments")

img_path = by_name[Path(info["file_name"]).name]       # lookup built earlier
a = np.array(Image.open(img_path).convert("L"))
m = make_mask(info)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(a, cmap="gray"); ax[0].set_title("image")
ax[1].imshow(a, cmap="gray")
ax[1].imshow(np.ma.masked_equal(m, 0), cmap="autumn", alpha=0.6); ax[1].set_title("with mask")
plt.show()
print(a.dtype, a.min(), a.max(), a.mean())

manifest = []
for info in data["images"]:
    p = by_name.get(Path(info["file_name"]).name)
    if p is not None:
        manifest.append((str(p), f"{out}/images/{info['id']}.png"))
json.dump(manifest, open(f"{out}/manifest.json", "w"))
print(len(manifest), "pairs of", len(data["images"]))