

import os, glob, csv, json, math
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
import matplotlib.pyplot as plt





# Preconditions (Step 1 paths)
# This can be avoided if you run the entire thing from the begining, but since two files are there, I have to make sure to point out the right message
assert "PATH_TRAIN" in globals() and os.path.isdir(PATH_TRAIN), "Define PATH_TRAIN in Step 1."
assert "PATH_TEST"  in globals() and os.path.isdir(PATH_TEST),  "Define PATH_TEST in Step 1."
os.makedirs("outputs/preds", exist_ok=True)
os.makedirs("outputs/curves", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[ENV]", device.type)

# Class order (match training)
# NOTE: Keras/Torch ImageFolder both use alpha-sorted folder names for class mapping

class_names = sorted([d for d in os.listdir(PATH_TRAIN) if os.path.isdir(os.path.join(PATH_TRAIN, d))])
idx_to_class = {i: c for i, c in enumerate(class_names)}
NUM_CLASSES  = len(class_names)
print("[INFO] Classes:", idx_to_class)







# Find best PyTorch checkpoint
def find_best_pt(prefer_tag="CNN_Variant_B_Improved"):
    patterns = [
        f"models/{prefer_tag}*best*.pt", f"models/{prefer_tag}*best*.pth",
        "models/*best*.pt", "models/*best*.pth",
        "models/*.pt", "models/*.pth",
    ]
    cands = []
    for pat in patterns: cands += glob.glob(pat)
    if not cands:
        raise FileNotFoundError("No PyTorch checkpoints (.pt/.pth) found in ./models/")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

BEST_CKPT_PATH = find_best_pt("CNN_Variant_B_Improved")
print("[CKPT] Using:", BEST_CKPT_PATH)







# Rebuild model & load weights
def build_effnet_b0(num_classes):
    m = efficientnet_b0(weights=None)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_feats, num_classes)
    return m

model = build_effnet_b0(NUM_CLASSES).to(device)
state = torch.load(BEST_CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# Preprocessing
# ToTensor() scales from [0,255] to [0,1] which matches “divide by 255"

IMG_SIZE = globals().get("IMG_SIZE", 256)
test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def load_tensor(path):
    img = Image.open(path).convert("RGB")
    return test_tfms(img).unsqueeze(0)




# FIGURE with PERCENTAGES
def annotate_all_probs(src_path, class_names, probs, out_path):
    img = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    header_h = max(60, img.height // 12)
    draw.rectangle([0, 0, img.width, header_h], fill=(0, 0, 0))
    order = np.argsort(-probs)
    y = 10
    for idx in order:
        draw.text((10, y), f"{class_names[idx]}: {probs[idx]*100:.2f}%", fill=(255,255,255))
        y += 18
    img.save(out_path)






# Required test images!
REQ = [
    ("test/crack/test_crack.jpg",              "crack"),
    ("test/missing-head/test_missinghead.jpg", "missing-head"),
    ("test/paint-off/test_paintoff.jpg",       "paint-off"),
]

print("\n[STEP 5 — Predictions on required images]")
saved_fig3 = []
for rel_path, expected in REQ:
    # Build absolute path inside PATH_TEST
    # NOTE: This might nt be necessarily of you just want to see the pictures here
    parts = rel_path.split("/")
    img_path = os.path.join(PATH_TEST, *parts[1:]) if parts and parts[0]=="test" else os.path.join(PATH_TEST, rel_path)
    if not os.path.exists(img_path):
        print(f"[WARN] Missing test image: {img_path} (expected: {expected})")
        continue

    xb = load_tensor(img_path).to(device)
    with torch.no_grad():
        logits = model(xb)
        probs = F.softmax(logits.float(), dim=1)[0].cpu().numpy()  # (C,)
    top = int(np.argmax(probs))
    print(f"{os.path.basename(img_path)} -> Pred: {idx_to_class[top]} ({probs[top]*100:.2f}%) | Expected: {expected}")

    out_path = os.path.join("outputs", "preds", f"fig3_{os.path.basename(img_path)}")
    annotate_all_probs(img_path, class_names, probs, out_path)
    saved_fig3.append(out_path)
    print(f"[OK] Saved Figure-3 style: {out_path}")

# Performance curves (if logs exist)
def latest(glob_pat):
    c = glob.glob(glob_pat)
    if not c: return None
    c.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return c[0]





acc_png  = "outputs/curves/figure2_accuracy.png"
loss_png = "outputs/curves/figure2_loss.png"

def plot_from_history_dict(hist, acc_path, loss_path, title="Model"):
    plt.figure(figsize=(7,5))
    plt.plot(hist.get("accuracy", []), label="Train Acc")
    plt.plot(hist.get("val_accuracy", []), label="Val Acc")
    plt.title(f"{title} — Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    plt.savefig(acc_path, bbox_inches="tight", dpi=150); plt.show()
    print("[INFO] Saved:", acc_path)

    plt.figure(figsize=(7,5))
    plt.plot(hist.get("loss", []), label="Train Loss")
    plt.plot(hist.get("val_loss", []), label="Val Loss")
    plt.title(f"{title} — Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.4); plt.legend()
    plt.savefig(loss_path, bbox_inches="tight", dpi=150); plt.show()
    print("[INFO] Saved:", loss_png)




# Try TF-style JSON first (if you followed earlier TF pipeline), else CSV (PyTorch logger)
hist_json = latest("outputs/histories/*_history.json")
if hist_json and os.path.exists(hist_json):
    with open(hist_json, "r") as f:
        hist = json.load(f)
    plot_from_history_dict(hist, acc_png, loss_png, title=os.path.basename(hist_json).replace("_history.json",""))
else:
    csv_path = latest("outputs/histories/*.csv")
    if csv_path and os.path.exists(csv_path):
        hist = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        with open(csv_path, "r", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                for k in hist.keys():
                    if k in row and row[k] not in ("", None):
                        try: hist[k].append(float(row[k]))
                        except: pass
        plot_from_history_dict(hist, acc_png, loss_png, title=os.path.basename(csv_path).replace(".csv",""))
    else:
        print("[WARN] No training history found in outputs/histories/. Skipping Figure-2.")



        #NOTE: Percentages are also shown here and the pictures are saved, but in the next cell I just brought them into the picture as preview








# Preview all saved images inline
import glob
from IPython.display import display, Image as IPyImage

for p in sorted(glob.glob("outputs/preds/fig3_*.jpg")):
    print(p)
    display(IPyImage(filename=p))









    #  Full test evaluation + artifacts



import os, csv, json, numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # if missing: pip install seaborn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert "PATH_TEST" in globals() and os.path.isdir(PATH_TEST), "Run Step 1 (PyTorch) first."

# Class order (match training)
train_tmp = datasets.ImageFolder(PATH_TRAIN)
class_to_idx = train_tmp.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}
NUM_CLASSES  = len(idx_to_class)

# Normalization
 # match Step 4


from torchvision.models import EfficientNet_B0_Weights
weights_enum = EfficientNet_B0_Weights.IMAGENET1K_V1
try:
    _tfms = weights_enum.transforms()
    _norms = [t for t in getattr(_tfms, "transforms", []) if isinstance(t, transforms.Normalize)]
    mean, std = (_norms[0].mean, _norms[0].std) if _norms else ((0.485,0.456,0.406),(0.229,0.224,0.225))
except Exception:
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)

IMG_SIZE = globals().get("IMG_SIZE", 256)
test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])




# Build test dataset/loader
test_ds = datasets.ImageFolder(PATH_TEST, transform=test_tfms)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True

)




# Rebuild model & load best ckpt
def build_effnet_b0(num_classes):
    m = efficientnet_b0(weights=None)
    in_feats = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_feats, num_classes)
    return m



best_path = "models/CNN_Variant_B_Improved_best.pt"
model = build_effnet_b0(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()



# Evaluate


all_y, all_pred, all_prob = [], [], []
all_paths = [p for (p, _) in test_ds.samples]  # absolute paths

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_y.extend(yb.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())




# Metrics


target_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
print("\n[TEST] Classification report:")
print(classification_report(all_y, all_pred, target_names=target_names, digits=4))

cm = confusion_matrix(all_y, all_pred)
os.makedirs("outputs/curves", exist_ok=True)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig("outputs/curves/test_confusion_matrix.png", dpi=150)
plt.show()
print('[INFO] Saved confusion matrix to "outputs/curves/test_confusion_matrix.png"')





# CSV of per image predictions

os.makedirs("outputs", exist_ok=True)
csv_path = "outputs/test_predictions.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    header = ["file", "true_label", "pred_label"] + [f"p_{c}" for c in target_names]
    w.writerow(header)
    for path, y, pred, prob in zip(all_paths, all_y, all_pred, all_prob):
        row = [os.path.basename(path), idx_to_class[y], idx_to_class[pred]] + list(map(lambda x: f"{x:.6f}", prob))
        w.writerow(row)
print(f'[INFO] Saved per-image predictions to "{csv_path}"')