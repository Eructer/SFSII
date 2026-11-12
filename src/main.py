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

# Local imports
from conv_net import UNet3
from util import (
    iou,
    print_param_names,
    lambda_params,
    check_lambda0,
    tiled_inference,
    evaluate,
    soft_iou_loss,
    loss_fn
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If you use loss_fn from util
# ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=DEVICE))

if DEVICE == "cpu":
    print("Using cpu")
    torch.set_num_threads(10)
    torch.set_num_interop_threads(2)
else:
    print("Using GPU")

# Train and Eval section below ↓

def train(model, train_data, val_data, epochs: int = 5):
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
            start = time.time()


            x, y = x.to(DEVICE), y.to(DEVICE)

            # Loss would need to be the correct segmentation one for comp
            with torch.autocast("cuda", dtype=torch.bfloat16):
              loss = F.cross_entropy(model(x), y)

            opt.zero_grad()

            loss.backward()

            opt.step()

            train_ind += 1

            end = time.time()

            print(f"\tTrain : {train_ind} | Time : {end-start:.3f} secs | Alloc {torch.cuda.max_memory_allocated()/1e9:.2f} GB", end='\r')


        val_loss = evaluate(model, val_data)

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

def test(model, test_loader):

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



def save_model(model, epoch):
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

def from_scratch(epochs: int, bs: int = 32, save: bool = False, training_size: int = 50, val_size: int = 50, kind = "sm", crop_size=16):
    """Function for getting the model from training

    Returns
    -------
        model (TinyUNet): Trained model

    """
    print(f"-----Getting model from training-----")

    # Create structure of data for lambda check
    # x = torch.randn(16, 1, 2048, 2048, device = DEVICE)

    # Create small conv, no IBNN. Baseline kinda
    # sm = TinyUNet(kind = "sm").to(DEVICE)
    # Create lite IBNN
    model = UNet3(kind = kind).to(DEVICE)

    # Full implicit layer
    # model = TinyUNet(kind = "full", ibnn_at=('enc1,)).to(DEVICE)

    # Print parameters
    print_param_names(model)
    # IBNN lambda check
    # check_lambda0(sm, model, x)

    # Solar Filament
    manifest = json.load(open("src/data/fil_modified/manifest.json"))
    # Discover files once (tile=None, so nothing is cropped here)
    random.Random(45).shuffle(manifest)
    n = len(manifest)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    train_pairs = manifest[:n_train]
    val_pairs   = manifest[n_train:n_train + n_val]
    test_pairs  = manifest[n_train + n_val:]

    train_data = FilamentDataset(train_pairs[:training_size], tile=512, crops_per_image=crop_size, augment=True)
    val_data   = FilamentDataset(val_pairs[:val_size], tile=None)
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
        save_model(model, f"{KIND}_{epochs}_{training_size}_{val_size}_{crop_size}")

    return model

def from_loaded(model_path : str, kind: str):
    """Function for getting model from saved file. The load function does load abritrary data be carefull

    Parameters
    ----------
        model_path (str): Path of model file

    Returns
    -------
        model (TinyUNet): Model loaded from file

    """
    # Load model, MUST USE SAME CLASS AS SAVED ONE
    model = UNet3(kind = kind).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
    return model

if __name__ == "__main__":
    # Free memory
    # torch.cuda.empty_cache()

    epochs = 10
    training_size = 100
    val_size = 50
    crop_size = 8
    KIND = "lite"

    # # Train and get model from scratch
    # model = from_scratch(epochs,kind=KIND, save=True, bs=1, training_size=training_size, val_size=val_size, crop_size=crop_size)

    # # Get model from file
    model = from_loaded('lite_5_100_50_8_model.pth', KIND)

    # torch.cuda.empty_cache()
    target_files = [
        "src/data/filament/test/test_images/20210730044930Lh.jpeg",
        "src/data/filament/test/test_images/20211119005210Mh.jpeg",
        "src/data/filament/test/test_images/20211214174010Mh.jpeg"
    ]
    ind = 0
    for file in target_files:
        arr, pred = segment_image(model, file, out_path=f"results/{ind}_{epochs}_{training_size}_{val_size}.png", compare=True, device=DEVICE, n_classes=4)
        ind += 1