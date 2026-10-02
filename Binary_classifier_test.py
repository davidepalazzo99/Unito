import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from scipy.ndimage import binary_dilation, binary_erosion

# Paths

TEST_FOLDER = "/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/test"

MASK_FOLDERS = ["/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/test/masks"]

CSV_PATH = "/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/test/0_metadata_filtrato_test.csv"

MODEL_PATH = "/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/chest_classifier_casual_bg.pth"

train_csv = pd.read_csv("/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train/0_metadata_filtrato.csv")

# Test mode: set to True to use lung masks

USE_MASK = True

df = pd.read_csv(CSV_PATH)

train_names = set(train_csv["filename"])
test_names = set(df["filename"])

common = train_names & test_names

print(f"Immagini comuni: {len(common)}")

if common:
    print(sorted(common)[:20])


TARGET_SIZE = (224, 224)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Binary classification labels

def normalize_finding(x):

    x = str(x).strip()

    if x in ["Pneumonia/Viral/COVID-19", "Pneumonia/Viral/SARS", "Pneumonia/Viral/MERS-CoV"]:
        return "COVID"

    return "NO_COVID"

df["label"] = df["finding"].apply(normalize_finding)

label_map = dict(zip(df["filename"], df["label"]))

# Classes distribution

print("\nDISTRIBUZIONE CLASSI:\n")
print(df["label"].value_counts())

# Only two classes

classes = ["NO_COVID", "COVID"]

class_to_idx = {c: i for i, c in enumerate(classes)}

print("\nCLASSI FINALI USATE:\n")
for k, v in class_to_idx.items():
    print(v, "->", k)


# Model

class ChestClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.densenet121(weights=None)
        print("Numero di input del classificatore:", self.backbone.classifier.in_features)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = ChestClassifier(len(classes)).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("MODELLO CARICATO")


# Test set construction

def collect_images(folder):

    files = []

    for f in os.listdir(folder):

        path = os.path.join(folder, f)

        if os.path.isfile(path):

            if f.lower().endswith((".png", ".jpg", ".jpeg")):

                files.append(path)

    return files

test_images = collect_images(TEST_FOLDER)

print(f"Totale immagini test: {len(test_images)}")


# Mask finder

def find_mask(stem):
    for folder in MASK_FOLDERS:
        for root, _, files in os.walk(folder):
            for f in files:
                if stem.lower() in f.lower() and "mask_post" in f.lower():
                    return os.path.join(root, f)
    return None

print(df["finding"].value_counts())

# Inference

y_true, y_pred, y_probs = [], [], []

missing_mask = 0
processed = 0

for i, img_path in enumerate(test_images):

    file = os.path.basename(img_path)
    stem = os.path.splitext(file)[0]

    
    # Original dataset -> CSV
    if file not in label_map:
        continue

    true_label = label_map[file]


    if true_label is None:
        continue
    
    
    if USE_MASK:
        mask_path = find_mask(stem)

        if mask_path is None:
            missing_mask += 1
            continue

    img = Image.open(img_path).convert("L").resize(TARGET_SIZE)

    img = np.array(img)

    if USE_MASK:

        mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
        mask = (np.array(mask) > 127).astype(np.uint8)

        lung_mean = img[mask == 1].mean() 
        img = np.where(mask == 1, img, lung_mean)

    # # Previous preprocessing using only masks with a black background
    # if USE_MASK:

    #     mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
    #     mask = (np.array(mask) > 127).astype(np.uint8)

    #     # Lung-only
    #     img = img * mask

    img = img.astype(np.float32) / 255.0

    x = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)


    with torch.no_grad(): 
        out = model(x)
        probs = torch.softmax(out, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
 
    y_true.append(true_label)
    y_pred.append(idx_to_class[pred])
    y_probs.append(probs.cpu().numpy()[0])

    processed += 1


# Debugging

print("\nVALID SAMPLES:", processed)
print("MISSING MASK:", missing_mask)

# Metrics

print("\nACCURACY:", accuracy_score(y_true, y_pred))

print("\nCLASSIFICATION REPORT") 
print(classification_report(y_true, y_pred, labels=classes))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred, labels=classes))


# ROC CURVE

y_true_binary = np.array([
    1 if label == "COVID" else 0
    for label in y_true
])

y_prob_covid = np.array(y_probs)[:, class_to_idx["COVID"]]

fpr, tpr, _ = roc_curve(y_true_binary, y_prob_covid)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))

plt.plot(fpr, tpr, label=f"COVID (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()


# SALIENCY MAP

#target_class
def compute_saliency(model, img_path, mask_path, target_class=None, use_mask=False):

    img = Image.open(img_path).convert("L").resize(TARGET_SIZE)
    img = np.array(img).astype(np.float32)

    #Mask
    if use_mask:
        mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
        mask = (np.array(mask) > 127).astype(np.float32)

        lung_mean = img[mask == 1].mean()
        img = np.where(mask == 1, img, lung_mean)
        # # Lung-only
        #img = img * mask

    img = img / 255.0

    x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    x = x.to(device)

    # Gradient
    x.requires_grad_()

    # Forward pass
    
    model.zero_grad()
  
    output = model(x)
    
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1).item()

    if target_class is None:
        target_class = predicted_class

    # Backpropagation

    score = output[0, target_class]
    score.backward()

    # Input gradient

    saliency = x.grad.abs()

    saliency, _ = torch.max(saliency, dim=1)

    saliency = (saliency.squeeze().cpu().detach().numpy())

    # Normalization

    saliency -= saliency.min()

    if saliency.max() != 0:
        saliency /= saliency.max()

    return (img, saliency, predicted_class,probabilities.detach().cpu().numpy()[0])


# Quantitative saliency analysis

def analyze_saliency_regions(saliency, mask, border_width=10):

    mask = mask.astype(bool)

    # Inner border

    eroded_mask = binary_erosion(mask, iterations=border_width)
    inner_border = mask & ~eroded_mask

    # Outer border

    dilated_mask = binary_dilation(mask, iterations=border_width)    
    outer_border = dilated_mask & ~mask

    # Lung core

    lung_core = eroded_mask

    # Background

    background = ~dilated_mask

    # Total saliency

    total_saliency = saliency.sum()

    # Region saliency

    saliency_core = saliency[lung_core].sum()
    saliency_inner_border = saliency[inner_border].sum()
    saliency_outer_border = saliency[outer_border].sum()
    saliency_background = saliency[background].sum()

    # Percentage dictionary
    
    results = {"lung_core": (saliency_core / total_saliency * 100), "inner_border": (saliency_inner_border / total_saliency * 100), "outer_border": (saliency_outer_border / total_saliency * 100), "background": (saliency_background / total_saliency * 100), "outside_lung": ((saliency_outer_border + saliency_background)/ total_saliency * 100)}
    

# Sum check

    print("Somma percentuali:", results["lung_core"] + results["inner_border"] + results["outer_border"] + results["background"])
   
    return results

# Saliency analysis for the test set

OUTPUT_DIR = os.path.join(os.path.dirname(MODEL_PATH), "saliency_normal")

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures_normal")

os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "regional_saliency_normal.csv")

results_csv = []

print("ANALISI SALIENCY DI TUTTO IL TEST SET")

for i, img_path in enumerate(test_images, start=1):

    filename = os.path.basename(img_path)
    stem = os.path.splitext(filename)[0]

    print(f"\n[{i}/{len(test_images)}] {filename}")


    # Mask finder

    mask_path = find_mask(stem)

    if mask_path is None:
        print("  ATTENZIONE: maschera non trovata")
        continue


    mask_img = Image.open(mask_path).convert("L")
    mask_img = mask_img.resize(TARGET_SIZE)

    mask = np.array(mask_img)
    mask = (mask > 127).astype(np.uint8)

    true_class = label_map[filename]


    # Saliency NO COVID

    (img, saliency_no_covid, _, probabilities) = compute_saliency(model, img_path, mask_path, target_class=0, use_mask=True)


    # Saliency COVID

    (img, saliency_covid, _, _) = compute_saliency(model, img_path, mask_path, target_class=1, use_mask=True)

    # Real prediction

    prob_no_covid = probabilities[0]
    prob_covid = probabilities[1]

    predicted_class = np.argmax(probabilities)

    predicted_class_name = classes[predicted_class]

    true_class_name = true_class

    # Regional analysis

    regions_no_covid = analyze_saliency_regions(saliency_no_covid, mask)

    regions_covid = analyze_saliency_regions(saliency_covid, mask)

    # Results

    print(f"  Vera classe: {true_class_name}")

    print(f"  Predetta: {predicted_class_name}")

    print(f"  P(NO_COVID) = {prob_no_covid:.4f}")

    print(f"  P(COVID)    = {prob_covid:.4f}")

    print(f"  NO_COVID - outside lung: {regions_no_covid['outside_lung']:.2f}%")

    print(f"  COVID    - outside lung: {regions_covid['outside_lung']:.2f}%")


    # Plots

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Preprocessed image

    axes[0].imshow(img, cmap="gray")

    axes[0].set_title(f"Originale\nTrue: {true_class_name}\nPred: {predicted_class_name}")

    axes[0].axis("off")

    # Saliency NO COVID

    axes[1].imshow(img, cmap="gray")

    im1 = axes[1].imshow(saliency_no_covid, cmap="jet", alpha=0.5, vmin=0, vmax=1)

    axes[1].set_title(f"Saliency NO_COVID\nOutside lung: {regions_no_covid['outside_lung']:.1f}%")

    axes[1].axis("off")

    # Saliency COVID

    axes[2].imshow(img, cmap="gray")

    im2 = axes[2].imshow(saliency_covid, cmap="jet", alpha=0.5, vmin=0, vmax=1)

    axes[2].set_title(f"Saliency COVID\nOutside lung: {regions_covid['outside_lung']:.1f}%")

    axes[2].axis("off")

    fig.colorbar(im2, ax=axes[2])


    fig.suptitle(filename, fontsize=12)

    plt.tight_layout()

    # Saving images

    output_figure = os.path.join(FIGURES_DIR, f"{stem}_saliency.png")

    plt.savefig(output_figure, dpi=150, bbox_inches="tight")

    plt.close(fig)


    # Saving data

    row = {"filename": filename,
           "true_class": true_class_name,
           "predicted_class": predicted_class_name,
           "prob_NO_COVID": prob_no_covid,
           "prob_COVID": prob_covid,
           # NO COVID
           "NO_COVID_lung_core": regions_no_covid["lung_core"],
           "NO_COVID_inner_border": regions_no_covid["inner_border"],
           "NO_COVID_outer_border": regions_no_covid["outer_border"],
           "NO_COVID_background": regions_no_covid["background"],
           "NO_COVID_outside_lung": regions_no_covid["outside_lung"],
           # COVID
           "COVID_lung_core": regions_covid["lung_core"],
           "COVID_inner_border": regions_covid["inner_border"],
           "COVID_outer_border": regions_covid["outer_border"],
           "COVID_background": regions_covid["background"],
           "COVID_outside_lung": regions_covid["outside_lung"]}

    results_csv.append(row)


# Save CSV file

saliency_df = pd.DataFrame(results_csv)

saliency_df.to_csv(CSV_PATH, index=False)

print("ANALISI SALIENCY COMPLETATA")

print(f"Immagini analizzate: {len(results_csv)}")

print(f"Figure salvate in: {FIGURES_DIR}")

print(f"CSV salvato in: {CSV_PATH}")