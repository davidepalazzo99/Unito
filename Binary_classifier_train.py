#MODEL FOR TRAINING
import os
import time
import copy
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from torch.utils.data import Dataset, DataLoader

# Paths

XRAY_FOLDER = "/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train"

MASK_FOLDER = "/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train/masks"

CSV_PATH = "/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train/0_metadata_filtrato.csv"

MODEL_SAVE_PATH = r"/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/chest_classifier_casual_bg.pth"

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

# Early stopping
PATIENCE = 10

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nDEVICE UTILIZZATO: {device}")

# Class for the dataset

class ChestDataset(Dataset):

    def __init__(self, df, img_dir, mask_dirs, class_to_idx, mask_mode="lung", randomize_background=False):

        self.df = df
        self.img_dir = img_dir
        self.mask_dirs = mask_dirs
        self.class_to_idx = class_to_idx
        self.mask_mode = mask_mode
        self.randomize_background = randomize_background

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
    
        filename = row["filename"]
        label = self.class_to_idx[row["label"]]
    
        #Check filename

        if os.path.isabs(filename):
        
            img_path = filename
        
        else:
        
            img_path = os.path.join(self.img_dir, filename)
    
        # Masks research

        img_stem = Path(filename).stem

        mask_path = None

        for current_mask_dir in self.mask_dirs:

            candidate = os.path.join(current_mask_dir, img_stem + "_mask_post.png")

            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            raise ValueError(
                f"Mask non trovata per {filename}. "
                f"Cercata: {img_stem}_mask_post.png"
            )
        
        # Image loading and convertion to grayscale
    
        try:
            img = Image.open(img_path).convert("L")
            mask = Image.open(mask_path).convert("L")
    
            img = np.array(img)
            mask = np.array(mask)

        
        except Exception as e:
            raise ValueError(f"Errore caricamento file {filename}: {e}")
    
        # Uniform resize for all images and masks.
    
        TARGET_SIZE = (224, 224)
        img = np.array(Image.fromarray(img).resize(TARGET_SIZE))
        mask = np.array(Image.fromarray(mask).resize(TARGET_SIZE))    
        
        # Dimension control

        if img.shape != mask.shape:
            raise ValueError(f"Mismatch dopo resize: {filename}")
    
        # Binary mask

        mask = (mask > 127).astype(np.uint8)
    
        # Mask application

        if self.mask_mode == "full":

            # Original (lung+outside)
            pass

        elif self.mask_mode == "lung":

            # lung-only
            #img = img * mask

            # Mean pixel intensity within the lungs
            lung_mean = img[mask == 1].mean()

            if self.randomize_background:

                # Random background
                random_background = random.randint(0, 255)

                img = np.where(mask == 1, img, random_background)

            else:

                # Background mean pixel intensity within the lungs
                img = np.where( mask == 1, img, lung_mean)

        elif self.mask_mode == "outside":

            # lung-outside
            img = img * (1 - mask)

        else:

            raise ValueError(f"mask_mode non valido: {self.mask_mode}. Usare 'full', 'lung' oppure 'outside'.")
    
     
        # Normalization
    
        img = img.astype(np.float32) / 255.0
        
      
        # Data augmentation
        
        if row["augmented"]:
        
            img = Image.fromarray((img * 255).astype(np.uint8))
        
            # Random rotation
            angle = random.uniform(-8, 8)
        
            img = TF.rotate( img, angle, interpolation=InterpolationMode.BILINEAR)
        
            # Translation
            translate_x = random.randint(-5, 5)
            translate_y = random.randint(-5, 5)
        
            img = TF.affine( img, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0, interpolation=InterpolationMode.BILINEAR)
        
            # Brightness variation
            brightness = random.uniform(0.9, 1.1)
        
            img = TF.adjust_brightness( img, brightness)
        
            # Contrast variation
            contrast = random.uniform(0.9, 1.1)
        
            img = TF.adjust_contrast(img, contrast)
        
            img = np.array(img).astype(np.float32) / 255.0
        
    
        # Tensor as input for densenet121
    
        img = torch.tensor(img).unsqueeze(0)
    
        # DenseNet -> 3 channels
        img = img.repeat(3, 1, 1)
    
        return img, torch.tensor(label, dtype=torch.long)

# Model class

class ChestClassifier(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = models.densenet121(weights="IMAGENET1K_V1")

    
        in_features = self.backbone.classifier.in_features

        # Final classifier replacement
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x): 

        return self.backbone(x)


# TRAIN LOOP

def train_one_epoch(model, loader, optimizer, criterion): 

    model.train()

    running_loss = 0

    for imgs, labels in loader: 

        imgs = imgs.to(device)

        labels = labels.to(device)

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item() 
    
    return running_loss / len(loader)


def validate(model, loader, criterion):

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


# Metadata reading

df = pd.read_csv(CSV_PATH)


# Binary finding

def normalize_finding(x):

    x = str(x).strip()

    if x in ["Pneumonia/Viral/COVID-19", "Pneumonia/Viral/SARS", "Pneumonia/Viral/MERS-CoV"]:
        return "COVID"

    return "NO_COVID"

df["label"] = df["finding"].apply(normalize_finding)


# Classes distribution

print("\nDISTRIBUZIONE CLASSI:\n")
print(df["label"].value_counts())

# Binary classes

classes = ["NO_COVID", "COVID"]

class_to_idx = {c: i for i, c in enumerate(classes)}

print("\nCLASSI FINALI USATE:\n")
for k, v in class_to_idx.items():
    print(v, "->", k)


# Train/validation split

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

splits = sgkf.split(df, y=df["label"], groups=df["patientid"])

train_idx, val_idx = next(splits)

train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()


# Training set balancing

print("\nDistribuzione TRAIN prima del bilanciamento:\n")
print(train_df["label"].value_counts()) 

target_size = len(train_df[train_df["label"] == "COVID"])

balanced_parts = []

for label in classes:

    subset = train_df[train_df["label"] == label].copy()

    subset["augmented"] = False

    balanced_parts.append(subset)

    missing = target_size - len(subset)

    if missing > 0:

        repetitions = missing // len(subset)
    
        remainder = missing % len(subset)

        for _ in range(repetitions):

            extra = subset.copy()

            extra["augmented"] = True

            balanced_parts.append(extra)

        if remainder > 0:

            extra = subset.sample(
                n=remainder,
                replace=False
            ).copy()

            extra["augmented"] = True

            balanced_parts.append(extra)

# Training set reconstruction
train_df = pd.concat(balanced_parts, ignore_index=True)

# Random shuffling of the entire training set
train_df = train_df.sample(frac=1).reset_index(drop=True)

val_df = val_df.copy()

val_df["augmented"] = False

# Final control after balancing
print("\nDistribuzione TRAIN dopo il bilanciamento:\n")
print(train_df["label"].value_counts())


# Training e validation dataset creation

train_dataset = ChestDataset(train_df.reset_index(drop=True), XRAY_FOLDER, [MASK_FOLDER], class_to_idx, mask_mode="lung", randomize_background=True)

val_dataset = ChestDataset(val_df.reset_index(drop=True), XRAY_FOLDER, [MASK_FOLDER], class_to_idx, mask_mode="lung", randomize_background=False)

# Dataloader

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  #non serve mescolare i dati per la validation

# MODEL

model = ChestClassifier(num_classes=len(classes)).to(device)

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"\nParametri totali:       {total_params:,}")
print(f"Parametri addestrabili: {trainable_params:,}")


# Loss function

criterion = nn.CrossEntropyLoss()

# OPTIMIZER: Adam

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training history

train_losses = []

val_losses = []

val_accuracies = []


# Training

print("\n=== TRAINING AVVIATO ===\n")

global_start = time.time()

#Early stopping

best_val_loss = np.inf

best_model_weights = copy.deepcopy(model.state_dict())

epochs_without_improvement = 0

for epoch in range(EPOCHS):

    epoch_start = time.time()

    # Train

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    # Validation

    val_loss, val_acc = validate( model, val_loader, criterion)

    # Evaluation metrics saving

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    val_accuracies.append(val_acc)

    # Times

    epoch_time = time.time() - epoch_start

    elapsed = time.time() - global_start

    remaining_epochs = EPOCHS - (epoch + 1)

    eta = remaining_epochs * epoch_time #ETA = Estimated Time of Arrival

    # Output epoch

    print(f"\n[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Epoch Time: {epoch_time:.1f}s | Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(model.state_dict())

        epochs_without_improvement = 0

        print(">> Miglior modello aggiornato")

    else:

        epochs_without_improvement += 1

        print(
            f">> Nessun miglioramento "
            f"({epochs_without_improvement}/{PATIENCE})"
        )

    if epochs_without_improvement >= PATIENCE:

        print("\n=== EARLY STOPPING ATTIVATO ===")

        break

# Restore the best model

model.load_state_dict(best_model_weights)

# Model saving

torch.save({

    "model_state_dict": model.state_dict(), #weights

    "class_to_idx": class_to_idx, #dictionary string -> index

    "classes": classes

}, MODEL_SAVE_PATH)

print(f"\nModello salvato in:\n{MODEL_SAVE_PATH}")

# Final graphs

epochs_range = range(1, len(train_losses) + 1)

# Loss plot (train e validation)

plt.figure(figsize=(10,5))

plt.plot(epochs_range, train_losses, label="Train Loss")

plt.plot(epochs_range, val_losses, label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training / Validation Loss")

plt.legend()

plt.grid(True)

plt.show()

# Accuracy

plt.figure(figsize=(10,5))

plt.plot(epochs_range, val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()