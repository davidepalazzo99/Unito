# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:13:56 2026

@author: Davide Palazzo
"""

import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize


# CONFIGURAZIONE
#=========================================================

TEST_FOLDER = "C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Test set"

MASK_FOLDERS = ["C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Test set/pred_20260729_01_53"]

CSV_PATH = "C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Test set/testlabels_clean.csv"

MODEL_PATH = "C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Cassettone/Seconda generazione/Terza generazione/Quarta generazione/Training 4/chest_classifier_background_augmented.pth"

# TEST PER VEDERE SE CI SONO IMMAGINI COMUNI FRA TEST E TRAINING SET
#=========================================================

train_csv = pd.read_csv("C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Training set/traininglabels_clean.csv")
test_csv = pd.read_csv(CSV_PATH)

train_names = set(train_csv["filename"].apply(os.path.basename)) 
test_names = set(test_csv["filename"].apply(os.path.basename))

common = train_names & test_names

print(f"Immagini comuni: {len(common)}")

if common:
    print(sorted(common)[:20])


#=========================================================
TARGET_SIZE = (224, 224)
device = "cuda" if torch.cuda.is_available() else "cpu"

df_csv = pd.read_csv(CSV_PATH)
 

def normalize_csv_label(x): # Normalizzazione 3 classi

    x = str(x).lower().strip()

    # Classe COVID
    if ("covid" in x or "sars" in x or "mers" in x):
        return "COVID"


    # Classe BACTERIAL
    if ("bacter" in x or "strepto" in x or "klebsiella" in x or "legionella" in x or "mycoplasma" in x or
        "e.coli" in x or "chlamydophila" in x or "tuberculosis" in x or "nocardia" in x):
        return "BACTERIA"


    # Classe VIRAL
    if ("virus" in x or "viral" in x or "influenza" in x or "varicella" in x or "herpes" in x):
        return "VIRAL"


    return None

df_csv["label"] = df_csv["label"].apply(normalize_csv_label)

# filename puliti
df_csv["filename"] = df_csv["filename"].apply(os.path.basename)

csv_label_map = dict(zip(df_csv["filename"], df_csv["label"]))


# CLASSI MODELLO
#=========================================================

classes = ["BACTERIA", "COVID", "VIRAL"]
idx_to_class = {i: c for i, c in enumerate(classes)}


# CARICAMENTO DEL MODELLO
#=========================================================

class ChestClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.densenet121(weights=None)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = ChestClassifier(len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("MODELLO CARICATO")


# COSTRUZIONE TEST SET 
#=========================================================

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


# RICERCA DELLE MASCHERE
#=========================================================

def find_mask(stem):
    for folder in MASK_FOLDERS:
        for root, _, files in os.walk(folder):
            for f in files:
                if stem.lower() in f.lower() and "mask" in f.lower():
                    return os.path.join(root, f)
    return None

print(df_csv["label"].value_counts())


# INFERENZA
#=========================================================

y_true, y_pred, y_probs = [], [], []

missing_img = 0
missing_mask = 0
processed = 0

for i, img_path in enumerate(test_images):

    file = os.path.basename(img_path)
    stem = os.path.splitext(file)[0]

    
    # LABEL
    #---------------------------------------------------------
    
    
    if file not in csv_label_map: # Riferimento a CSV
        continue
    
    true_label = csv_label_map[file]
    
    if true_label is None:
        continue

    mask_path = find_mask(stem)
    if mask_path is None:
        missing_mask += 1
        continue

    img = Image.open(img_path).convert("L").resize(TARGET_SIZE)
    mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)

    img = np.array(img)
    mask = (np.array(mask) > 127).astype(np.uint8)

    
    # APPLICAZIONE MASCHERE
    #---------------------------------------------------------

    lung_mean = img[mask == 1].mean()

    img = np.where(mask == 1, img, lung_mean)

    img = img.astype(np.float32) / 255.0

    x = torch.tensor(img).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)

        pred = torch.argmax(probs, dim=1).item()

    y_true.append(true_label)
    y_pred.append(idx_to_class[pred])
    y_probs.append(probs.cpu().numpy()[0])

    processed += 1


# OUTPUT DEBUG
#=========================================================

print("\nVALID SAMPLES:", processed)
print("MISSING MASK:", missing_mask)


# METRICHE DI PERFORMANCE
#=========================================================

print("\nACCURACY:", accuracy_score(y_true, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_true, y_pred, labels=classes))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred, labels=classes))


# ROC CURVE
#=========================================================

y_true_bin = label_binarize(y_true, classes=classes)
y_probs = np.array(y_probs)

plt.figure()

for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=c)

plt.plot([0,1],[0,1],'--')
plt.legend()
plt.title("ROC Curve")
plt.show()



# SALIENCY MAP
#=========================================================

def compute_saliency(model, img_path, mask_path, target_class=None):

    
    # 1 CARICAMENTO IMMAGINE E MASCHERE
    #---------------------------------------------------------

    img = Image.open(img_path).convert("L").resize(TARGET_SIZE)
    mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)

    img = np.array(img).astype(np.float32)
    mask = (np.array(mask) > 127).astype(np.float32)

    
    # 2 APPLICAZIONE DELLA MASCHERA
    #---------------------------------------------------------

    lung_mean = img[mask == 1].mean()

    img_masked = np.where(mask == 1, img, lung_mean)

    img_masked = img_masked / 255.0

    
    # 3 CONVERSIONE IN TENSORE
    #---------------------------------------------------------

    x = torch.tensor(img_masked, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    x = x.to(device)
    x.requires_grad_()

    
    # 4 FORWARD PASS
    #---------------------------------------------------------

    model.zero_grad()

    output = model(x)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(output, dim=1).item()

    if target_class is None:
        target_class = predicted_class

    
    # 5 BACKPROPAGATION
    #---------------------------------------------------------

    score = output[0, target_class]

    score.backward()

    
    # 6 GRADIENTE DELL'INPUT
    #---------------------------------------------------------

    saliency = x.grad.abs()

    saliency, _ = torch.max(saliency, dim=1)

    saliency = (saliency.squeeze().cpu().detach().numpy())

    
    # 7 NORMALIZZAZIONE
    #---------------------------------------------------------

    saliency -= saliency.min()

    if saliency.max() != 0:
        saliency /= saliency.max()

    return (img_masked, saliency, predicted_class, probabilities.detach().cpu().numpy()[0])


# TEST DI ABLATION DEL BACKGROUND
#=========================================================

def background_ablation_test(model, img_path, mask_path):
    img = Image.open(img_path).convert("L").resize(TARGET_SIZE)
    mask = Image.open(mask_path).convert("L").resize(TARGET_SIZE)

    img = np.array(img).astype(np.float32)
    mask = (np.array(mask) > 127).astype(np.float32)

    
    img = img / 255.0 # Normalizzazione dell'immagine

    
    lung_mean = img[mask == 1].mean() # Media dei pixel all'interno dei polmoni

    
    img_mean = np.where(mask == 1, img, lung_mean) # Tre versioni: stesso polmone, background diverso
    img_black = np.where(mask == 1, img, 0.0)
    img_gray = np.where(mask == 1, img, 0.5)

    def to_tensor(image):
        return (torch.tensor(image, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device))

    x_mean = to_tensor(img_mean)
    x_black = to_tensor(img_black)
    x_gray = to_tensor(img_gray)

    with torch.no_grad():
        out_mean = model(x_mean)
        out_black = model(x_black)
        out_gray = model(x_gray)

        prob_mean = torch.softmax(out_mean, dim=1)[0].cpu().numpy()
        prob_black = torch.softmax(out_black, dim=1)[0].cpu().numpy()
        prob_gray = torch.softmax(out_gray, dim=1)[0].cpu().numpy()

    return prob_mean, prob_black, prob_gray


# ANALISI QUANTITATIVA DELLA SALIENCY (PERCENTUALI)
#=========================================================

from scipy.ndimage import binary_dilation, binary_erosion


def analyze_saliency_regions(saliency, mask, border_width=10):

    mask = mask.astype(bool)

    
    # BORDO INTERNO DEL POLMONE
    #---------------------------------------------------------

    eroded_mask = binary_erosion(mask, iterations=border_width)

    inner_border = mask & ~eroded_mask

    
    # BORDO ESTERNO DEL POLMONE
    #---------------------------------------------------------

    dilated_mask = binary_dilation(mask, iterations=border_width)

    outer_border = dilated_mask & ~mask

    
    # INTERNO PROFONDO DEL POLMONE
    #---------------------------------------------------------

    lung_core = eroded_mask

    
    # RESTO DEL BACKGROUND
    #---------------------------------------------------------

    background = ~dilated_mask

    
    # SALIENCY TOTALE
    #---------------------------------------------------------

    total_saliency = saliency.sum()

    
    # SALIENCY NELLE REGIONI
    #---------------------------------------------------------

    saliency_core = saliency[lung_core].sum()
    saliency_inner_border = saliency[inner_border].sum()
    saliency_outer_border = saliency[outer_border].sum()
    saliency_background = saliency[background].sum()

    
    # PERCENTUALI
    #---------------------------------------------------------

    results = {"lung_core": (saliency_core / total_saliency * 100),

        "inner_border": (saliency_inner_border / total_saliency * 100),

        "outer_border": (saliency_outer_border / total_saliency * 100),

        "background": (saliency_background / total_saliency * 100)}

    return results


# CARICAMENTO MASCHERA PER ANALISI REGIONALE
#=========================================================

mask_array = np.array(Image.open(mask_path).convert("L").resize(TARGET_SIZE))

mask_array = mask_array > 127


# SELEZIONE IMMAGINE SINGOLA
#=========================================================

IMAGE_NAME = "000002-11-a.jpg"

img_path = os.path.join(
    TEST_FOLDER,
    IMAGE_NAME
)

if not os.path.exists(img_path):
    raise FileNotFoundError(
        f"Immagine non trovata:\n{img_path}"
    )


# RICERCA MASCHERA
#=========================================================

stem = os.path.splitext(IMAGE_NAME)[0]

mask_path = find_mask(stem)

if mask_path is None:
    raise FileNotFoundError(f"Mask non trovata per:\n{IMAGE_NAME}")





print("\n======================================")
print("SALIENCY MAP")
print("======================================")

print("Immagine:", IMAGE_NAME)
print("Mask:", mask_path)



# CALCOLO DELLE SALIENCY
#=========================================================


class_indices = {"BACTERIA": 0, "COVID": 1, "VIRAL": 2}


results = {}

for class_name, class_index in class_indices.items():

    (img_masked, saliency, predicted_class, probabilities) = compute_saliency(model, img_path, mask_path, target_class=class_index)

    results[class_name] = {"saliency": saliency, "probabilities": probabilities}



# INFORMAZIONI SULLA PREDIZIONE
#=========================================================

print("\n======================================")
print("SALIENCY MAP")
print("======================================")

print("Immagine:", IMAGE_NAME)
print("Mask:", mask_path)

print("\nProbabilità del modello:")

for i, c in enumerate(classes):
    print(f"{c}: "
        f"{results[c]['probabilities'][i]:.4f}")


# ANALISI QUANTITATIVA
#=========================================================

print("\n======================================")
print("ANALISI REGIONALE DELLA SALIENCY")
print("======================================")

for class_name in class_indices:

    saliency = results[class_name]["saliency"]

    regional_results = analyze_saliency_regions(saliency, mask_array, border_width=10)

    print(f"\n{class_name}")

    print(f"  Polmone interno: "
        f"{regional_results['lung_core']:.2f}%")

    print(f"  Bordo interno: "
        f"{regional_results['inner_border']:.2f}%")

    print(f"  Bordo esterno: "
        f"{regional_results['outer_border']:.2f}%")

    print(f"  Background: "
        f"{regional_results['background']:.2f}%")


# VISUALIZZAZIONE
#=========================================================

plt.figure(figsize=(16, 5))


# INPUT
#---------------------------------------------------------

plt.subplot(1, 4, 1)

plt.imshow( img_masked, cmap="gray")

plt.axis("off")
plt.title("Input mascherato")



# BACTERIA
#---------------------------------------------------------

plt.subplot(1, 4, 2)

plt.imshow(img_masked, cmap="gray")

plt.imshow(results["BACTERIA"]["saliency"], cmap="jet", alpha=0.5)

plt.axis("off")
plt.title("Saliency BACTERIA")



# COVID
#---------------------------------------------------------

plt.subplot(1, 4, 3)

plt.imshow(img_masked, cmap="gray")

plt.imshow(results["COVID"]["saliency"], cmap="jet", alpha=0.5)

plt.axis("off")
plt.title("Saliency COVID")



# VIRAL
#---------------------------------------------------------

plt.subplot(1, 4, 4)

plt.imshow(img_masked, cmap="gray")

plt.imshow(results["VIRAL"]["saliency"], cmap="jet", alpha=0.5)

plt.axis("off")
plt.title("Saliency VIRAL")


plt.tight_layout()
plt.show()


# IMMAGINI BACTERIA CLASSIFICATE COME COVID ERRONEAMENTE
#=========================================================

covid_images = ["SARS-10.1148rg.242035193-g04mr34g0-Fig8a-day0.jpeg", "SARS-10.1148rg.242035193-g04mr34g0-Fig8b-day5.jpeg",
    "SARS-10.1148rg.242035193-g04mr34g0-Fig8c-day10.jpeg", "SARS-10.1148rg.242035193-g04mr34g05x-Fig5-day9.jpeg",
    "SARS-10.1148rg.242035193-g04mr34g09a-Fig9a-day17.jpeg"]



# ESECUZIONE TEST
#=========================================================


covid_index = classes.index("COVID")

results_ablation = []

for image_name in covid_images:

    img_path = os.path.join(TEST_FOLDER, image_name)

    
    base_name = os.path.splitext(image_name)[0]

    mask_path = None

    for folder in MASK_FOLDERS:
        candidate = os.path.join(folder, base_name + "_mask.png")

        if os.path.exists(candidate):
            mask_path = candidate
            break

    if mask_path is None:
        print("MASK NON TROVATA:", image_name)
        continue

    prob_mean, prob_black, prob_gray = background_ablation_test(model, img_path, mask_path)

    covid_mean = prob_mean[covid_index]
    covid_black = prob_black[covid_index]
    covid_gray = prob_gray[covid_index]

    results_ablation.append({"image": image_name, "covid_mean": covid_mean, "covid_black": covid_black,
        "covid_gray": covid_gray})

    print("\n", image_name)
    print("P(COVID) background media :", covid_mean)
    print("P(COVID) background nero  :", covid_black)
    print("P(COVID) background grigio:", covid_gray)

    print("Delta nero - media :", covid_black - covid_mean)
    print("Delta grigio - media:", covid_gray - covid_mean)