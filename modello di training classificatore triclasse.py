# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:47:05 2026

@author: Davide Palazzo
"""

import os
import time
import copy
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from torch.utils.data import Dataset, DataLoader, random_split

#CONFIGURAZIONE
#=========================================================

XRAY_FOLDER = "C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Training set"

MASK_FOLDER = "C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Training set/pred_20260728_18_01"

CSV_PATH = "C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Dataset ordinato/Terza generazione/Training set/traininglabels_clean.csv"

#Dove salvare il modello finale

MODEL_SAVE_PATH = r"C:/Users/Davide Palazzo/Desktop/L'Abisso/Università/Data Analysis for Machine Learning/Progetto/Cassettone/Seconda generazione/Terza generazione/Quarta generazione/Training 5/chest_classifier_background_augmented.pth"

#Iper parametri

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

#Early stopping

PATIENCE = 10

#Divisione Train-Validation

TRAIN_SPLIT = 0.8

#Device

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDEVICE UTILIZZATO: {device}")

#DATASET CUSTOM
#=========================================================

class ChestDataset(Dataset):
    def __init__(self, df, img_dir, mask_dirs, class_to_idx, randomize_background=False):

        self.df = df
        self.img_dir = img_dir
        self.mask_dirs = mask_dirs
        self.class_to_idx = class_to_idx
        self.randomize_background = randomize_background

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        filename = row["filename"]
        label = self.class_to_idx[row["label"]]

        # Vecchio dataset => filename relativo
        # Nuovo dataset => path assoluto

        if os.path.isabs(filename):

            img_path = filename

        else:

            img_path = os.path.join(
                self.img_dir,
                filename
            )

        from pathlib import Path
        base = Path(filename).stem

        
        # RICERCA MASCHERE
        #=========================================================

        extensions = [
            ".png",
            ".PNG",
            ".jpg",
            ".JPG",
            ".jpeg",
            ".JPEG"
        ]

        mask_path = None
        img_stem = base

        
        # CERCA IN TUTTE LE CARTELLE LE MASCHERE
        #=========================================================

        for current_mask_dir in self.mask_dirs:

            
            # PROVA 1: cerca prima "_mask_post"
            #---------------------------------------------------------

            for ext in extensions:

                candidate = os.path.join(current_mask_dir, img_stem + "_mask_post" + ext)

                if os.path.exists(candidate):

                    mask_path = candidate
                    break

            
            # PROVA 2: se "_mask_post" non esiste, cerca "_mask"
            #---------------------------------------------------------

            if mask_path is None:

                for ext in extensions:

                    candidate = os.path.join(current_mask_dir, img_stem + "_mask" + ext)

                    if os.path.exists(candidate):

                        mask_path = candidate
                        break

            
            # Se abbiamo trovato una maschera, usciamo dal ciclo
            #=========================================================

            if mask_path is not None:
                break

        
        # STAMPA ERRORE SE NON È STATA TROVATA NESSUNA MASCHERA
        #=========================================================

        if mask_path is None:

            raise ValueError(f"Nessuna mask ('_mask_post' o '_mask') trovata per {filename}")

        
        # CARICAMENTO IMMAGINI
        #=========================================================

        try:

            img = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")

            img = np.array(img)
            mask = np.array(mask)

        except Exception as e:

            raise ValueError(f"Errore caricamento file {filename}: {e}")

        
        # RIDIMENSIONAMENTO IMMAGINE 224 X 224
        #=========================================================

        TARGET_SIZE = (224, 224)

        img = np.array(
            Image.fromarray(img).resize(TARGET_SIZE)
        )

        mask = np.array(
            Image.fromarray(mask).resize(TARGET_SIZE)
        )

        
        # CONTROLLO CHE LE DIMENSIONI DELL'IMMAGINE E DELLA MASCHERA COMBACINO
        #=========================================================

        if img.shape != mask.shape:

            raise ValueError(f"Mismatch dopo resize: {filename}")

        
        # BINARIZZAZIONE DELLA MASCHERA
        #=========================================================

        mask = (mask > 127).astype(np.uint8)

        
        # CALCOLO DEL BACKGROUND COME MEDIA ALL'INTERNO DEI POLMONI
        #=========================================================

        lung_mean = img[mask == 1].mean() # Valore medio dei pixel all'interno dei polmoni

        
        # BACKGROUND RANDOMIZZATO TRA 0 E 255
        #=========================================================

        if self.randomize_background:

            # Training:
                # Il background viene scelto casualmente
                # In questo modo il modello non può associare una determinata classe ad uno specifico valore
                # del background

            background_value = random.randint(0, 255)

        else:

            # Validation:
                # comportamento deterministico.
            background_value = lung_mean

        
        # APPLICAZIONE MASCHERE
        #=========================================================

        img = np.where(mask == 1, img, background_value) # Si pone l'immagine originale dove la maschera è
                                                            # bianca, altrimenti si mette il background

        
        # NORMALIZZAZIONE RISULTATO
        #=========================================================

        img = img.astype(np.float32) / 255.0

        
        # DATA AUGMENTATION SOLO PER LE COPIE
        #=========================================================

        if row["augmented"]:

            img = Image.fromarray((img * 255).astype(np.uint8))

            
            # Rotazione casuale
            #---------------------------------------------------------

            angle = random.uniform(-8, 8)

            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)

            
            # Piccola traslazione
            #---------------------------------------------------------

            translate_x = random.randint(-5, 5)
            translate_y = random.randint(-5, 5)

            img = TF.affine(img, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0, interpolation=InterpolationMode.BILINEAR)

            
            # Leggera variazione luminosità
            #---------------------------------------------------------

            brightness = random.uniform(0.9, 1.1)

            img = TF.adjust_brightness(img, brightness)

            
            # Leggera variazione contrasto
            #---------------------------------------------------------

            contrast = random.uniform(0.9, 1.1)

            img = TF.adjust_contrast(img, contrast)

            img = np.array(img).astype(np.float32) / 255.0

        
        # TENSORE
        #=========================================================

        img = torch.tensor(img).unsqueeze(0)

        
        img = img.repeat(3, 1, 1) # DenseNet -> 3 canali

        return img, torch.tensor(label, dtype=torch.long)


# MODELLO PER IL TRANSFER LEARNING
#=========================================================

class ChestClassifier(nn.Module):
    def __init__(self, num_classes):

        super().__init__()

        
        self.backbone = models.densenet121(weights="IMAGENET1K_V1") # Richiamo DENSENET121

        
        in_features = self.backbone.classifier.in_features # Numero feature ultimo layer

        
        self.backbone.classifier = nn.Linear(in_features, num_classes) # Sostituzione classificatore finale
                                                                        # Viene applicata una relazione lineare
    def forward(self, x):

        return self.backbone(x)


# LOOP DI TRAINING
#=========================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    running_loss = 0

    for imgs, labels in loader:

        imgs = imgs.to(device) 
                                    # Spostamento elementi su CPU
        labels = labels.to(device)

        
        outputs = model(imgs) # Forward

        
        loss = criterion(outputs, labels) # Loss

        
        optimizer.zero_grad() # Reset gradienti

        
        loss.backward() # Backpropagation

        
        optimizer.step() # Aggiornamento pesi

        running_loss += loss.item()

    return running_loss / len(loader)


# VALIDATION
#=========================================================

def validate(model, loader, criterion):
    model.eval()

    running_loss = 0

    correct = 0

    total = 0

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)

            labels = labels.to(device)

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

    avg_loss = running_loss / len(loader)

    accuracy = correct / total

    return avg_loss, accuracy


# LETTURA DATASET
#=========================================================

df = pd.read_csv(CSV_PATH)

print("\nLABEL ORIGINALI CSV:")
print(df["label"].value_counts())


# NORMALIZZAZIONE LABEL DA ORIGINALI A 3 CLASSI
#=========================================================

def normalize_label(x):
    x = str(x).lower().strip()

    # classe COVID
    if ("covid" in x or "sars" in x or "mers" in x):

        return "COVID"

    # classe BACTERIAL
    if ("bacter" in x or "strepto" in x or "klebsiella" in x or "legionella" in x or "mycoplasma" in x or
        "e.coli" in x or "chlamydophila" in x or "tubercolosis" in x or "nocardia" in x):

        return "BACTERIA"

    # classe VIRAL
    if ("virus" in x or "viral" in x or "influenza" in x or "varicella" in x or "herpes" in x):

        return "VIRAL"

    return None

df["label"] = df["label"].apply(normalize_label)

print("\nLABEL DOPO NORMALIZZAZIONE:")
print(df["label"].value_counts(dropna=False))



df = df.dropna(subset=["label"]) #elimina classi inutili

print("\nDISTRIBUZIONE CLASSI:\n")
print(df["label"].value_counts())


# FORZA 3 CLASSI
#=========================================================

classes = ["BACTERIA", "COVID", "VIRAL"]

class_to_idx = {c: i for i, c in enumerate(classes)}

print("\nCLASSI FINALI USATE:\n")

for k, v in class_to_idx.items():
    print(v, "->", k)


# TRAIN / VALIDATION SPLIT 
#=========================================================

train_df, val_df = train_test_split(df, test_size=1 - TRAIN_SPLIT, stratify=df["label"],
random_state=42, shuffle=True)


# BILANCIAMENTO TRAINING SET
#=========================================================

print("\nDistribuzione TRAIN "
"prima del bilanciamento:\n")

print(train_df["label"].value_counts())

target_size = len(train_df[train_df["label"] == "COVID"])

balanced_parts = []

for label in classes:
    subset = train_df[train_df["label"] == label].copy()

    subset["augmented"] = False

    balanced_parts.append(subset)

    missing = (target_size - len(subset))

    if missing > 0:

        
        repetitions = (missing // len(subset)) # Ripetizioni uniformi

        remainder = (missing % len(subset))

        for _ in range(repetitions):

            extra = subset.copy()

            extra["augmented"] = True

            balanced_parts.append(extra)

        if remainder > 0:

            extra = subset.sample(n=remainder, replace=False).copy()

            extra["augmented"] = True

            balanced_parts.append(extra)

train_df = pd.concat(balanced_parts, ignore_index=True)

train_df = train_df.sample(frac=1).reset_index(drop=True)

val_df = val_df.copy()

val_df["augmented"] = False

print("\nDistribuzione TRAIN "
"dopo il bilanciamento:\n")

print(train_df["label"].value_counts())


# CREAZIONE DATASET
#=========================================================

# MEMENTO IMPORTANTE:
# Il training ha un background casuale
# Mentre il validation ha il background originale

train_dataset = ChestDataset(train_df.reset_index(drop=True), XRAY_FOLDER, [MASK_FOLDER], class_to_idx,
randomize_background=True)

val_dataset = ChestDataset(val_df.reset_index(drop=True), XRAY_FOLDER, [MASK_FOLDER], class_to_idx,
randomize_background=False)


# DATALOADER
#=========================================================

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# MODELLO
#=========================================================

model = ChestClassifier(num_classes=len(classes)).to(device)

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nParametri totali:       {total_params:,}")
print(f"Parametri addestrabili: {trainable_params:,}")


# DISTRIBUZIONE CLASSI
#=========================================================

print("\nDISTRIBUZIONE CLASSI:\n")

print(df["label"].value_counts())


# LOSS FUNCTION
#=========================================================

criterion = nn.CrossEntropyLoss()


# INTRODUZIONE ADAM COME OTTIMIZZATORE
#=========================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# EARLY STOPPING
#=========================================================

best_val_loss = np.inf

best_model_weights = copy.deepcopy(model.state_dict())

epochs_without_improvement = 0


# STORICO METRICHE
#=========================================================

train_losses = []

val_losses = []

val_accuracies = []


# AVVIO DEL TRAINING
#=========================================================

print("\n=== TRAINING AVVIATO ===\n")

global_start = time.time()

for epoch in range(EPOCHS):

    epoch_start = time.time()

    
    # TRAIN
    #---------------------------------------------------------

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    
    # VALIDATION
    #---------------------------------------------------------

    val_loss, val_acc = validate(model, val_loader,criterion)

    
    # SALVATAGGIO METRICHE
    #---------------------------------------------------------

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    
    # TEMPI
    #---------------------------------------------------------

    epoch_time = (time.time() - epoch_start)

    elapsed = (time.time() - global_start)

    remaining_epochs = (EPOCHS - (epoch + 1))

    eta = (remaining_epochs * epoch_time)

    
    # OUTPUT EPOCHE
    #---------------------------------------------------------

    print(f"\n[Epoch {epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Epoch Time: {epoch_time:.1f}s | "
        f"Elapsed: {elapsed/60:.1f}m | "
        f"ETA: {eta/60:.1f}m")

    
    # EARLY STOPPING
    #---------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(model.state_dict())

        epochs_without_improvement = 0

        print(">> Miglior modello aggiornato")

    else:

        epochs_without_improvement += 1

        print(
            f">> Nessun miglioramento "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})")

        if epochs_without_improvement >= PATIENCE:

            print("\n=== EARLY STOPPING "
                "ATTIVATO ===")

            break



# RIPRISTINO MIGLIOR MODELLO
#=========================================================

model.load_state_dict(best_model_weights)


# RIPRISTINO MIGLIOR MODELLO
#=========================================================

model.load_state_dict(best_model_weights)


# SALVATAGGIO MODELLO
#=========================================================

torch.save({"model_state_dict": model.state_dict(), "class_to_idx": class_to_idx,
"classes": classes}, MODEL_SAVE_PATH)

print(
f"\nModello salvato in:\n"
f"{MODEL_SAVE_PATH}")


# GRAFICI FINALI
#=========================================================

epochs_range = range(1, len(train_losses) + 1)


# LOSS
#=========================================================

plt.figure(figsize=(10, 5))

plt.plot(epochs_range, train_losses, label="Train Loss")

plt.plot(epochs_range, val_losses, label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training / Validation Loss")

plt.legend()

plt.grid(True)

plt.show()


#ACCURACY
#=========================================================

plt.figure(figsize=(10, 5))

plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()

