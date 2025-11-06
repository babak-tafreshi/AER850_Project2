# Setting up the environment based on COLAB T4 GPU
# if you wanna run this on CPU (not suggested) you have to change the environment in step 3 and 4, but the results here are based on this
#UPDATE: Tensor here is checked but I later switched to pytorch because I ran out of Colab GPU :)))))

import os, pathlib, json, math, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)



#================================================================================
#STEP 1
#Sidenote: I'll be re-importing the libraries becasue during my work I don't save the model everytime so it doesn't cache

import os, shutil, json, pathlib, random, math
from collections import Counter
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms






# THIS IS THE PATH DEFINITION SECTION
# YOU MIGHT WANNA CHANGE IT BASED ON WHAT AND WHERE YOU HAVE SAVED THE DATA
DRIVE_TRAIN = "/content/drive/MyDrive/AER 850/Project 2/train"
DRIVE_VAL   = "/content/drive/MyDrive/AER 850/Project 2/valid"
DRIVE_TEST  = "/content/drive/MyDrive/AER 850/Project 2/test"









# Copy to local SSD (just to make it fast to train)
def copy_to_local(src, dst):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Missing: {src}")
    if not dst.exists():
        print(f"[DATA] Copying {src} -> {dst} ...")
        shutil.copytree(src, dst)
    else:
        print(f"[DATA] Using cached local copy: {dst}")
    return str(dst)

PATH_TRAIN = copy_to_local(DRIVE_TRAIN, "/content/dataset/train")
PATH_VAL   = copy_to_local(DRIVE_VAL,   "/content/dataset/val")
PATH_TEST  = copy_to_local(DRIVE_TEST,  "/content/dataset/test")

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manualSeed = torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False  # allow perf
torch.backends.cudnn.benchmark = True       # speed on fixed image sizes




# CORE PARAM!!!!!

IMG_SIZE   = 256
BATCH_SIZE = 64     # Starting with this

#UPDATE: worked out



NUM_WORKERS= 4
PIN_MEMORY = True

# Transforms
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.06,0.06))], p=0.8),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    transforms.ToTensor(),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])






# Dataset based on the paths!
train_ds = datasets.ImageFolder(PATH_TRAIN, transform=train_tfms)
val_ds   = datasets.ImageFolder(PATH_VAL,   transform=eval_tfms)
test_ds  = datasets.ImageFolder(PATH_TEST,  transform=eval_tfms)

class_to_idx = train_ds.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
NUM_CLASSES  = len(class_to_idx)

print("[INFO] Classes:", class_to_idx)






# Counts an weights
def count_images_per_class(root):
    counts = Counter()
    for cls_name, idx in datasets.ImageFolder(root).class_to_idx.items():
        cls_dir = pathlib.Path(root)/cls_name
        counts[cls_name] = len([p for p in cls_dir.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".gif",".tiff",".webp")])
    return counts

train_counts = count_images_per_class(PATH_TRAIN)
total = sum(train_counts.values())
weights = {class_to_idx[c]: total / (NUM_CLASSES * max(1, n)) for c, n in train_counts.items()}
print("[INFO] Train counts:", dict(train_counts))
print("[INFO] Class weights (for loss):", weights)






# and the data loader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS>0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS>0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=NUM_WORKERS>0)





# Saving up the label map
os.makedirs("outputs", exist_ok=True)
with open("outputs/label_map.json","w") as f:
    json.dump(class_to_idx, f, indent=2)
print('[INFO] Saved "outputs/label_map.json"')





# Export globals for later steps
globals().update(dict(
    PATH_TRAIN=PATH_TRAIN, PATH_VAL=PATH_VAL, PATH_TEST=PATH_TEST,
    IMG_SIZE=IMG_SIZE, NUM_CLASSES=NUM_CLASSES, class_to_idx=class_to_idx, idx_to_class=idx_to_class,
    BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, PIN_MEMORY=PIN_MEMORY,
    train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, class_weights=weights
))






#STEP 2 (PyTorch): CNN Architectures
# Baseline A (ReLU)
# Improved B (BN + LeakyReLU + ELU head + L2)


import torch
import torch.nn as nn
import torch.nn.functional as F

USE_GAP = True

def _conv_block(in_ch, out_ch, use_bn=False, use_lrelu=False):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.1, inplace=True) if use_lrelu else nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)







class CNN_Variant_A_Baseline(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, use_bn=False, use_lrelu=False),
            _conv_block(32, 64, use_bn=False, use_lrelu=False),
            _conv_block(64,128, use_bn=False, use_lrelu=False),
            _conv_block(128,256, use_bn=False, use_lrelu=False)
        )
        if USE_GAP:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        else:
            # Extra pool (16 -> 8) to control params before flatten
            self.preflat = nn.MaxPool2d(2)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*8*8, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )






    def forward(self, x):
        x = self.features(x)
        if USE_GAP:
            return self.head(x)
        x = self.preflat(x)
        return self.head(x)




class CNN_Variant_B_Improved(nn.Module):
    def __init__(self, num_classes=3, l2=1e-4):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3,  64, use_bn=True, use_lrelu=True),
            _conv_block(64,128, use_bn=True, use_lrelu=True),
            _conv_block(128,256, use_bn=True, use_lrelu=True),
            _conv_block(256,256, use_bn=True, use_lrelu=True),
        )
        if USE_GAP:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(256, 256), nn.ELU(inplace=True),
                nn.BatchNorm1d(256), nn.Dropout(0.6),
                nn.Linear(256, num_classes)
            )
        else:
            self.preflat = nn.MaxPool2d(2)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256*8*8, 256), nn.ELU(inplace=True),
                nn.BatchNorm1d(256), nn.Dropout(0.6),
                nn.Linear(256, num_classes)
            )
        self.l2 = l2

    def forward(self, x):
        x = self.features(x)
        if USE_GAP:
            return self.head(x)
        x = self.preflat(x)
        return self.head(x)











# Quick param counts
#Last versions used to have up to 64m parameters, now it reduced to 1m !!!!!!




def count_params(m):
    return sum(p.numel() for p in m.parameters())

model_a = CNN_Variant_A_Baseline(num_classes=NUM_CLASSES)
model_b = CNN_Variant_B_Improved(num_classes=NUM_CLASSES)

print(f"[INFO] A params: {count_params(model_a):,}")
print(f"[INFO] B params: {count_params(model_b):,}")

# Export builders
globals().update(dict(CNN_Variant_A_Baseline=CNN_Variant_A_Baseline,
                      CNN_Variant_B_Improved=CNN_Variant_B_Improved))

































# STEP 3 (PyTorch): Hyperparameter candidates


import pandas as pd

HP_TRIALS = [
    {"name":"A_relu_adam1e-3",         "model":"A", "optimizer":"adam",    "lr":1e-3,  "label_smooth":0.00, "weight_decay":0.0},
    {"name":"B_lrelu_elu_adam1e-3_l2", "model":"B", "optimizer":"adam",    "lr":1e-3,  "label_smooth":0.00, "weight_decay":1e-4},
    {"name":"A_relu_sgd2e-2",          "model":"A", "optimizer":"sgd",     "lr":2e-2,  "label_smooth":0.00, "weight_decay":0.0},
    {"name":"B_lrelu_rms5e-4",         "model":"B", "optimizer":"rmsprop", "lr":5e-4,  "label_smooth":0.00, "weight_decay":0.0},
]

rows = []
for hp in HP_TRIALS:
    m = CNN_Variant_B_Improved(NUM_CLASSES) if hp["model"]=="B" else CNN_Variant_A_Baseline(NUM_CLASSES)
    rows.append({"name":hp["name"], "model":hp["model"], "optimizer":hp["optimizer"],
                 "lr":hp["lr"], "label_smooth":hp["label_smooth"],
                 "weight_decay":hp["weight_decay"], "params":sum(p.numel() for p in m.parameters())})

df = pd.DataFrame(rows)
print(df)




# Export



import os, json
os.makedirs("outputs", exist_ok=True)
with open("outputs/hparam_candidates_pytorch.json","w") as f:
    json.dump(HP_TRIALS, f, indent=2)
globals().update(dict(HP_TRIALS=HP_TRIALS))













# STEP 4

# UPDATE: Removes 'verbose' from ReduceLROnPlateau (older PT builds)

# Uses NUM_WORKERS = 2 (per Colab warning)


import os, time, json, numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[ENV]", device)





# defaults from Step 1 (override if missing)
IMG_SIZE    = globals().get("IMG_SIZE", 256)
BATCH_SIZE  = globals().get("BATCH_SIZE", 64)
NUM_WORKERS = 2   # I'm gonna keep this two because of colab worker warning
PIN_MEMORY  = globals().get("PIN_MEMORY", True)

assert "PATH_TRAIN" in globals() and "PATH_VAL" in globals(), "Run Step 1 first."
assert "NUM_CLASSES" in globals(), "Run Step 1 first."





# ImageNet normalization
from torchvision.models import EfficientNet_B0_Weights
weights_enum = EfficientNet_B0_Weights.IMAGENET1K_V1
try:
    _tfms = weights_enum.transforms()
    _norms = [t for t in getattr(_tfms, "transforms", []) if isinstance(t, transforms.Normalize)]
    if _norms:
        mean, std = _norms[0].mean, _norms[0].std
    else:
        raise RuntimeError("Normalize not in weights transforms")
except Exception:
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.06,0.06))], p=0.8),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.ImageFolder(PATH_TRAIN, transform=train_tfms)
val_ds   = datasets.ImageFolder(PATH_VAL,   transform=eval_tfms)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=NUM_WORKERS > 0
)









# class weights
def count_images_per_class(root):
    tmp = datasets.ImageFolder(root)
    counts = Counter()
    for cname, idx in tmp.class_to_idx.items():
        cdir = os.path.join(root, cname)
        n = sum(1 for r,_,files in os.walk(cdir)
                for f in files if os.path.splitext(f)[1].lower() in (".jpg",".jpeg",".png",".bmp",".gif",".tiff",".webp"))
        counts[cname] = n
    return counts

class_to_idx_local = train_ds.class_to_idx
idx_to_class_local = {v:k for k,v in class_to_idx_local.items()}
NUM_CLASSES_LOCAL  = len(class_to_idx_local)
assert NUM_CLASSES_LOCAL == NUM_CLASSES, "Class count changed between steps."

train_counts = count_images_per_class(PATH_TRAIN)
total = sum(train_counts.values())
weights = {class_to_idx_local[c]: total / (NUM_CLASSES_LOCAL * max(1, n)) for c, n in train_counts.items()}
w = torch.tensor([weights[i] for i in range(NUM_CLASSES_LOCAL)], dtype=torch.float32, device=device)
print("[DATA] Class weights:", {idx_to_class_local[i]: float(w[i].cpu().numpy()) for i in range(NUM_CLASSES_LOCAL)})









# MODELLLLL
#UPDATE: IT WORKED
from torchvision.models import efficientnet_b0
def build_effnet_b0(num_classes):
    m = efficientnet_b0(weights=weights_enum)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feats, num_classes)
    return m

model = build_effnet_b0(NUM_CLASSES_LOCAL).to(device)




# speed knobs
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")





# optim/sched/loss
lr_head = 7.5e-4
optimizer = optim.Adam(model.parameters(), lr=lr_head, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # <- no 'verbose'
scaler = GradScaler(enabled=torch.cuda.is_available())

label_smooth = 0.05
criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smooth).to(device)




# loops
def run_epoch(loader, train=True):
    model.train(train)
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(imgs)
            loss = criterion(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, correct/total

def evaluate_preds(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

def save_ckpt(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_ckpt(path):
    model.load_state_dict(torch.load(path, map_location=device))




# phase 1: train head
#UPDATE: ACC UPTO 90
for p in model.features.parameters():    p.requires_grad = False
for p in model.classifier.parameters():  p.requires_grad = True

EPOCHS, PATIENCE = 20, 4
best_val_acc, best_path = 0.0, "models/CNN_Variant_B_Improved_best.pt"
history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
no_improve = 0

print("\n================ PHASE 1 (head only) ================")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    with torch.no_grad():
        vl_loss, vl_acc = run_epoch(val_loader, train=False)
    scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={vl_loss:.4f} acc={vl_acc:.4f} | "
          f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc; save_ckpt(best_path); no_improve = 0
        print(f"  [CKPT] Improved val_acc -> {best_val_acc:.4f}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("  [EARLY STOP] No improvement. Stopping Phase 1.")
            break







# phase 2: fine-tune
#UPDATE: last 80%



print("\n================ PHASE 2 (fine-tune) ================")
load_ckpt(best_path)

feat_params = list(model.features.parameters())
cut = int(len(feat_params) * 0.2)  # unfreeze last 80%
for i, p in enumerate(feat_params):
    p.requires_grad = (i >= cut)

for g in optimizer.param_groups:
    g['lr'] = 3e-4

no_improve = 0
for epoch in range(1, max(6, PATIENCE+2)+1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    with torch.no_grad():
        vl_loss, vl_acc = run_epoch(val_loader, train=False)
    scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)

    print(f"[FT] Epoch {epoch:02d} | "
          f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={vl_loss:.4f} acc={vl_acc:.4f} | "
          f"lr={optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc; save_ckpt(best_path); no_improve = 0
        print(f"  [CKPT] Improved val_acc -> {best_val_acc:.4f}")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("  [EARLY STOP] No improvement. Stopping FT.")
            break

print(f"\n[RESULT] Best val_acc: {best_val_acc:.4f}")











#  plots and classification report
#UPDATE: ACCURACY OF 94% and RECALL 95 % for paint off !!!!


from sklearn.metrics import classification_report, confusion_matrix
load_ckpt(best_path)
y_true, y_pred = evaluate_preds(val_loader)
print("\n[VAL] Classification report:")
print(classification_report(y_true, y_pred, target_names=[idx_to_class_local[i] for i in range(NUM_CLASSES_LOCAL)]))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

os.makedirs("outputs/curves", exist_ok=True)
plt.figure(figsize=(7,5))
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Acc")
plt.grid(True, ls="--", alpha=0.4); plt.legend()
plt.savefig("outputs/curves/effnet_acc.png", dpi=150, bbox_inches="tight"); plt.show()

plt.figure(figsize=(7,5))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.grid(True, ls="--", alpha=0.4); plt.legend()
plt.savefig("outputs/curves/effnet_loss.png", dpi=150, bbox_inches="tight"); plt.show()










