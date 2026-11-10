import time
from collections import defaultdict
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# For mnist
import torchvision
import torchvision.transforms as transforms
import json

# IBNN imports
from modified_rf.nn_layers import SMLayer, IBNNLiteLayer, IBNNLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    print("Using cpu")
    torch.set_num_threads(10)
    torch.set_num_interop_threads(2)
else:
    print("Using GPU")

# MNIST setup, using MNIST to test with images
class MNISTSegmentationDataset(Dataset):
    def __init__(self, root, max_per_folder = None, shuffle_pick = False, threshold=127, seed = 0):
        root = Path(root)
        all_paths = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in [".jpg"]
        )

        by_folder = defaultdict(list)
        for p in all_paths:
            by_folder[p.parent].append(p)

        rng = random.Random(seed)
        self.paths = []
        for folder in sorted(by_folder):
            files = by_folder[folder]
            if max_per_folder is not None:
                if shuffle_pick:
                    files = rng.sample(files, min(max_per_folder, len(files)))
                else:
                    files = files[:max_per_folder]
            self.paths.extend(files)

        self.threshold = threshold

    def __getitem__(self, index):
        path = self.paths[index]

        image = np.array(Image.open(path).convert("L"))
        image = torch.from_numpy(image)

        # The image is the mask
        # For every pixel in the image, if the pixel is > 127 put 1, else put 0. Tensor of 1's and 0's
        mask = (image > self.threshold).long()
        image = (image.float() / 255.0).unsqueeze(0)

        return image, mask

    def __len__(self):
        return len(self.paths)

class FilamentDataset(Dataset):
    def __init__(self, pairs, tile=512, crops_per_image=16, fg_prob=0.5, augment=False):
        """
        pairs: list of (image_path, mask_path)
        tile:  crop size; None returns the full image (validation/test)
        """
        self.pairs = list(pairs)
        self.tile = tile
        self.crops_per_image = crops_per_image if tile else 1
        self.fg_prob = fg_prob
        self.augment = augment

    def __len__(self):
        return len(self.pairs) * self.crops_per_image

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index // self.crops_per_image]
        img = np.array(Image.open(img_path).convert("L"))
        mask = (np.array(Image.open(mask_path)) > 0).astype(np.int64)

        if self.tile:
            H, W = mask.shape
            t = self.tile
            y = x = None
            if random.random() < self.fg_prob:
                fg = np.flatnonzero(mask)
                if fg.size:
                    cy, cx = divmod(int(fg[random.randrange(fg.size)]), W)
                    y = min(max(cy - t // 2, 0), H - t)
                    x = min(max(cx - t // 2, 0), W - t)
            if y is None:
                y, x = random.randint(0, H - t), random.randint(0, W - t)
            img = img[y:y + t, x:x + t]
            mask = mask[y:y + t, x:x + t]

        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)

        image = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).unsqueeze(0)
        return image, torch.from_numpy(np.ascontiguousarray(mask))

# Binary segmentation
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

class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 1, n_classes: int = 2, kind: str = "sm", ibnn_at: tuple = ("enc1"), width: int = 16):
        super().__init__()
        k = lambda name: kind if name in ibnn_at else "sm"
        self.enc1 = Block(k("enc1"), in_ch, width)
        self.enc2 = Block(k("enc2"), width, 2 * width)
        self.enc3 = Block(k("enc3"), 2*width, 4*width)
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

def print_param_names(model: TinyUNet):
    """Function for printing out model parameters

    Parameters
    ----------
        model (TinyUNet): Model

    """
    # Check
    for n, p in model.named_parameters():
        print(f"  {n:60s} {tuple(p.shape)} requires_grad={p.requires_grad}")

def lambda_params(model: TinyUNet) -> dict:
    """Function for getting lambda params

    Parameters
    ----------
        model (TinyUNet): Model

    Returns
    -------
        dict : Model lambda params

    """
    return {n : p for n, p in model.named_parameters() if "lambda" in n.lower()}

def check_lambda0(sm_model: TinyUNet, ibnn_model: TinyUNet, x: torch.Tensor):
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

def tiled_inference(model, img, tile_size=512, overlap=64):
    H, W = img.shape[1], img.shape[2]
    out = torch.zeros_lik(img)
    counts = torch.zeros_like(img)
    for y in range(0, H - overlap, tile_size - overlap):
        for x in range(0, W - overlap, tile_size - overlap):
            y2 = min(y + tile_size, H)
            x_2 = min(x + tile_size, W)
            tile = img[:, y:y2, x:x2]
            out[:, y:y2, x:x2] == model(tile.unsqueeze(0))[0]
            counts[:, y:y2, x:x2] += 1
    return out / counts.clamp(min = 1)

# Train and Eval section below ↓

def train(model, train_data, val_data, epochs: int = 5) -> TinyUNet:
    """Function for training a model

    Parameters
    ----------
        model (TinyUNet): Model
        train_data (Dataloader): Training data
        test_data (Dataloader): Testing/Validating data
        epochs (int): Epochs. Default '5'

    Returns
    -------
        model (TinyUNet): Trained model
    """
    # Check
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\n-----Training-----\n")
    t0 = time.time()

    for ep in range(epochs):
        print(f"Epoch : {ep}", end='\n')

        model.train()

        train_ind = 0

        for x, y in train_data:

            print(f"\tTrain : {train_ind} | Alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB", end='\r')

            x, y = x.to(DEVICE), y.to(DEVICE)

            # Loss would need to be the correct segmentation one for comp
            with torch.autocast("cuda", dtype=torch.bfloat16):
              loss = F.cross_entropy(model(x), y)

            opt.zero_grad()

            loss.backward()

            opt.step()

            train_ind += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in val_data:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += iou(model(x), y)

        val_loss /= len(val_data)

        torch.cuda.empty_cache()

            # scores = [iou(model(x.to(DEVICE)), y.to(DEVICE)) for x, y in test]

        print(f"epoch {ep}  loss {loss.item():.3f}  val IoU {val_loss:.4f} | Mem {torch.cuda.memory_allocated()/1e9:.2f}")

    secs = time.time() - t0

    # Print train time
    # Took longer than 1 minute to train
    tstr = ""
    if secs > 60:
        tstr = f"Train time (min) : {secs//60}"
    else:
        tstr = f"Train time (min): {secs}"
    print(tstr)


    mem = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else float("nan")

    for n, p in lambda_params(model).items():
        print(f"    {n} : mean = {p.mean().item():.4f}")

    return model

def test(model: TinyUNet, test_loader) -> TinyUNet:

    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            test_loss += iou(pred, y)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.numel()
    print(f"Test Loss : {test_loss/len(test_loader):.4f} | Accuracy : {(correct/total):.2f}")

    return model



def save_model(model: TinyUNet, epoch):
    """Function for saving model

    Parameters
    ----------
        model (TinyUNet): Model to save
    """
    torch.save(model.state_dict(), f"{epoch}_model.pth")

# Change sizes depending on image
def segment_image(model, image_path, out_path="segmented.png", device="cpu", n_classes = 1, compare = False) -> np.array and np.array:
    """Function for model predicting/segmenting

    Parameters
    ----------
        model (TinyUNet): Conv Model
        image_path (str): Input image path
        out_path (str): Segmented image path. Default 'segmented.png'
        device (str): Device to use. Default 'cpu'
        n_classes (int): Prediction classes, based on loss function. Default '1'
        compare (bool): Compare input and segmented image by plot. Default 'False'

    Returns
    -------
        arr (np.array): Input image
        pred (np.array): Predicted segmentation
    """
    # Eval model
    model.eval()

    # Get input image data
    img = Image.open(image_path).convert("L").resize((2048,2048))
    # Turn image into array
    arr = np.array(img)

    # Use if white on black
    # arr = 255 - arr

    # Turn input image into torch array
    x = torch.from_numpy(arr).float().div(255.0)
    # Squeeze input image
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    # Prediction
    if n_classes == 1:
        pred = (torch.sigmoid(out) > 0.5).squeeze().cpu().numpy()
    else:
        pred = out.argmax(dim=1).squeeze().cpu().numpy()
        pred = pred > 0 if n_classes == 2 else pred


    # Create mask image
    mask_img = Image.fromarray((pred * 255).astype(np.uint8)) if pred.dtype == bool else Image.fromarray((pred * (255 // max(pred.max(), 1))).astype(np.uint8))

    # Save prediction image
    mask_img.save(out_path)

    # Compare input and predicted by plotting
    if compare:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(arr, cmap="gray")
        ax[0].set_title("Input")

        ax[1].imshow(pred, cmap="gray")
        ax[1].set_title("Prediction Mask")
        for a in ax:
            a.axis("off")
        plt.savefig(out_path)

    return arr, pred

def from_scratch(epochs: int, bs: int = 32, save: bool = False) -> TinyUNet:
    """Function for getting the model from training

    Returns
    -------
        model (TinyUNet): Trained model

    """
    print(f"-----Getting model from training-----")

    # Create structure of data for lambda check
    # x = torch.randn(16, 1, 2048, 2048, device = DEVICE)

    # Create small conv, no IBNN. Baseline kinda
    sm = TinyUNet(kind = "sm").to(DEVICE)

    # Create lite IBNN
    model = TinyUNet(kind = "lite", ibnn_at=("mid",)).to(DEVICE)

    # Full implicit layer
    # model = TinyUNet(kind = "full", ibnn_at=('enc1,)).to(DEVICE)

    # Print parameters
    print_param_names(model)
    # IBNN lambda check
    # check_lambda0(sm, model, x)

    # -------------------------------------------------------------
    # Setup and load MNIST data
    # https://github.com/teavanist/MNIST-JPG/blob/master/MNIST%20Dataset%20JPG%20format.zip
    # train_path = "src/data/MNIST/training"
    # test_path = "src/data/MNIST/testing"
    # MNIST
    # train_data = MNISTSegmentationDataset(train_path, max_per_folder=sample_size)
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # Solar Filament
    manifest = json.load(open("src/data/fil_modified/manifest.json"))
    # Discover files once (tile=None, so nothing is cropped here)
    random.Random(45).shuffle(manifest)
    n = len(manifest)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    train_pairs = manifest[:n_train]
    val_pairs   = manifest[n_train:n_train + n_val]
    test_pairs  = manifest[n_train + n_val:]

    train_data = FilamentDataset(train_pairs, tile=512, crops_per_image=16, augment=True)
    val_data   = FilamentDataset(val_pairs, tile=None)
    test_data  = FilamentDataset(test_pairs, tile=None)

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_data, batch_size=1, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_data, batch_size=1, num_workers=2, pin_memory=True)


    # Train model
    model = train(model, train_loader, val_loader, epochs)

    # Test model
    model = test(model, test_loader)

    # Save model
    if save:
        print(f"-----Saving Model-----")
        save_model(model, f"{epochs}")

    return model

def from_loaded(model_path : str) -> TinyUNet:
    """Function for getting model from saved file. The load function does load abritrary data be carefull

    Parameters
    ----------
        model_path (str): Path of model file

    Returns
    -------
        model (TinyUNet): Model loaded from file

    """
    # Load model, MUST USE SAME CLASS AS SAVED ONE
    model = TinyUNet(kind = 'lite', ibnn_at=("enc1",)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    return model

if __name__ == "__main__":
    # Free memory
    torch.cuda.empty_cache()

    epochs = 5

    # Train and get model from scratch
    model = from_scratch(epochs, save=True, bs=1)

    # Get model from file
    # model = from_loaded('model.pth')

    torch.cuda.empty_cache()

    # Prediction array and pred
    target_file = "src/data/filament/test/test_images/20110424064534Lh.jpeg"
    arr, pred = segment_image(model, target_file, out_path=f"results/{epochs}.png", compare=True, device=DEVICE, n_classes=4)

    # print(arr)
    # print(pred)
