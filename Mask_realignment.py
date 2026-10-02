import os
import csv
import numpy as np
from PIL import Image
from scipy.ndimage import sobel, binary_erosion
import matplotlib.pyplot as plt


# Paths

IMAGE_FOLDER = ("/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train")

MASK_FOLDER = ("/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train/masks")

OUTPUT_FOLDER = ("/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train/masks_riallineate")

CONTROL_FOLDER = ("/home/pc-gruppo-physmed-unito/Desktop/Progetto_DA_covid/train/controllo_riallineamento")

# Parameters

TARGET_SIZE = (224, 224)
N_IMAGES = 591
MAX_SHIFT = 20
MIN_RELATIVE_IMPROVEMENT = 0.05

#  Folders creation

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CONTROL_FOLDER, exist_ok=True)


# List with all the images

image_files = []

for f in os.listdir(IMAGE_FOLDER):

    if f.lower().endswith((".png", ".jpg", ".jpeg")):

        full_path = os.path.join(IMAGE_FOLDER, f)

        # Evita eventuali cartelle
        if os.path.isfile(full_path):
            image_files.append(f)

image_files.sort()

image_files = image_files[:N_IMAGES]

print("ANALISI AUTOMATICA DEL RIALLINEAMENTO DELLE MASCHERE")

print(f"Numero di immagini analizzate: {len(image_files)}")
print(f"Dimensione analisi: {TARGET_SIZE}")
print(f"Massimo shift: ±{MAX_SHIFT} pixel")
print(f"Miglioramento minimo richiesto: "
    f"{MIN_RELATIVE_IMPROVEMENT * 100:.1f}%")

print()

# Masks search

def find_mask(stem):

    for root, _, files in os.walk(MASK_FOLDER):

        for f in files:

            if (stem.lower() in f.lower() and "mask_post" in f.lower()):

                return os.path.join(root, f)

    return None


# Masks translation

def shift_mask(mask, dx, dy):

    """
    Trasla una maschera di dx pixel orizzontalmente
    e dy pixel verticalmente.

    dx > 0  -> destra
    dx < 0  -> sinistra

    dy > 0  -> basso
    dy < 0  -> alto
    """

    shifted = np.zeros_like(mask)

    h, w = mask.shape

    # Inizial coordinates

    x1_src = max(0, -dx)

    if dx >= 0:
        x2_src = w - dx
    else:
        x2_src = w

    y1_src = max(0, -dy)

    if dy >= 0:
        y2_src = h - dy
    else:
        y2_src = h

    # Final coordinates

    x1_dst = max(0, dx)
    x2_dst = x1_dst + (x2_src - x1_src)
    y1_dst = max(0, dy)
    y2_dst = y1_dst + (y2_src - y1_src)

    # Copy in the new mask

    if x2_dst > x1_dst and y2_dst > y1_dst:
        
        shifted[y1_dst:y2_dst, x1_dst:x2_dst] = mask[y1_src:y2_src, x1_src:x2_src]

    return shifted

# Mask Boundary

def get_mask_boundary(mask):

    """
    Estrae il bordo della maschera.
    """

    mask = mask.astype(bool)

    eroded = binary_erosion(mask)

    boundary = mask & ~eroded

    return boundary


# Image gradient calculate

def calculate_gradient(image):

    """
    Calcola il modulo del gradiente dell'immagine tramite Sobel

    Le zone con forti variazioni di intensità
    avranno valori di gradiente elevati.
    """

    image = image.astype(np.float32)
    gx = sobel(image, axis=1) 
    gy = sobel(image, axis=0)
    gradient = np.sqrt(gx**2 + gy**2)

    return gradient


# Score calculation

def calculate_score(mask, gradient):

    """
    Calcola quanto il bordo della maschera
    coincide con zone ad alto gradiente.

    """

    boundary = get_mask_boundary(mask)

    if boundary.sum() == 0:
        return 0

    score = gradient[boundary].mean()

    return score


# Analysis

results_summary = []


for i, image_name in enumerate(image_files):


    print(f"[{i + 1}/{len(image_files)}] "
        f"{image_name}")

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    stem = os.path.splitext(image_name)[0]

    mask_path = find_mask(stem)

    if mask_path is None:

        print("ATTENZIONE: maschera non trovata")

        results_summary.append({"image": image_name, "mask_found": False, "original_score": "", "best_score": "", "relative_improvement": "", "dx_224": "", "dy_224": "", "dx_original": "", "dy_original": "", "correction_applied": False})

        continue

    print(f"Maschera trovata: {os.path.basename(mask_path)}")


    image_original = Image.open(image_path).convert("L")
    original_size = image_original.size

    print(f"Dimensione immagine originale: {original_size}")

    mask_original = Image.open(mask_path).convert("L")
    mask_original_array = np.array(mask_original)
    mask_original_binary = (mask_original_array > 127)

    print(f"Dimensione maschera originale: {mask_original_array.shape[::-1]}")


    image = image_original.resize(TARGET_SIZE, Image.Resampling.BILINEAR) #Vedere se specificare, fai una prova senza anche per la maschera
    image = np.array(image, dtype=np.float32)

    mask = mask_original.resize(TARGET_SIZE, Image.Resampling.NEAREST)
    mask = np.array(mask) > 127


    # Gradient calculation

    gradient = calculate_gradient(image)


    # Original score masks

    original_score = calculate_score(mask, gradient)

    print(f"Score iniziale: {original_score:.4f}")

    # Best shift research

    best_score = original_score
    best_dx = 0
    best_dy = 0

    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):

        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):

            shifted_mask = shift_mask(mask, dx, dy)

            score = calculate_score(shifted_mask, gradient)

            if score > best_score:

                best_score = score

                best_dx = dx
                best_dy = dy

    print(f"Miglior shift trovato: dx = {best_dx}, dy = {best_dy}")

    print(f"Miglior score: {best_score:.4f}")

    # Relative improvement calculation

    if original_score != 0:

        relative_improvement = ((best_score - original_score) / original_score)

    else:

        relative_improvement = 0

    print(f"Miglioramento relativo: {relative_improvement * 100:.2f}%")

    # Decision: apply or not the translation

    if (relative_improvement >= MIN_RELATIVE_IMPROVEMENT):

        correction_applied = True

        final_dx = best_dx
        final_dy = best_dy

        print("CORREZIONE APPLICATA")

    else:

        correction_applied = False

        final_dx = 0
        final_dy = 0

        print("Correzione NON applicata: miglioramento insufficiente")

    original_width = original_size[0]
    original_height = original_size[1]

    target_width = TARGET_SIZE[0]
    target_height = TARGET_SIZE[1]

    scale_x = (original_width / target_width)
    scale_y = (original_height / target_height)

    dx_original = round(final_dx * scale_x)
    dy_original = round(final_dy * scale_y)

    print(f"Shift nella dimensione originale: dx = {dx_original}, dy = {dy_original}")

    # Translation application

    mask_corrected_original = shift_mask(mask_original_binary, dx_original, dy_original)

    mask_corrected_uint8 = (mask_corrected_original.astype(np.uint8) * 255)
    mask_corrected_image = Image.fromarray(mask_corrected_uint8)

    # Save corrected mask

    output_mask_path = os.path.join(OUTPUT_FOLDER, os.path.basename(mask_path))

    mask_corrected_image.save(output_mask_path)

    print(f"Maschera salvata in:\n {output_mask_path}")

    # Visual control

    image_display = np.array(image_original)

    mask_before = (mask_original_array > 127)

    mask_after = (np.array(mask_corrected_image)> 127)

    plt.figure(figsize=(15, 5))

    # Original image

    plt.subplot(1, 3, 1)

    plt.imshow(image_display, cmap="gray")

    plt.axis("off")

    plt.title("Immagine originale")

    # Overlay image and mask before translation

    plt.subplot(1, 3, 2)

    plt.imshow(image_display, cmap="gray")

    plt.contour(mask_before, levels=[0.5], linewidths=1)

    plt.axis("off")

    plt.title("Prima\ndx=0, dy=0")

    # Overlay image and mask after translation

    plt.subplot(1, 3, 3)
    plt.imshow(image_display, cmap="gray")
    plt.contour(mask_after, levels=[0.5], linewidths=1)
    plt.axis("off")
    plt.title(f"Dopo\ndx={dx_original}, dy={dy_original}")
    plt.tight_layout()

    # Nome del file di controllo
    control_name = (os.path.splitext(image_name)[0] + "_mask_post.png")

    control_path = os.path.join(CONTROL_FOLDER, control_name)

    plt.savefig(control_path, dpi=150, bbox_inches="tight")

    plt.close()

    # Save results

    results_summary.append({"image": image_name,
                            "mask_found": True,
                            "original_score": original_score,
                            "best_score": best_score,
                            "relative_improvement": relative_improvement,
                            "dx_224": best_dx,
                            "dy_224": best_dy,
                            "dx_original": dx_original,
                            "dy_original": dy_original,
                            "correction_applied": correction_applied})

# Save CSV file

csv_path = os.path.join(CONTROL_FOLDER, "risultati_riallineamento.csv")

fieldnames = ["image", "mask_found", "original_score", "best_score", "relative_improvement", "dx_224", "dy_224", "dx_original", "dy_original",  "correction_applied"]

with open(csv_path, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    writer.writerows(results_summary)


# Final riepilogy

n_found = sum(r["mask_found"] for r in results_summary)

n_corrected = sum(r["correction_applied"] for r in results_summary if r["mask_found"])

n_not_corrected = (n_found - n_corrected)

print()
print("ANALISI COMPLETATA")

print(f"Immagini analizzate: {len(image_files)}")

print(f"Maschere trovate: {n_found}")

print(f"Maschere corrette: {n_corrected}")

print(f"Maschere lasciate invariate: {n_not_corrected}")

print(f"Maschere riallineate:\n{OUTPUT_FOLDER}")

print(f"Controlli visivi:\n{CONTROL_FOLDER}")

print(f"Risultati CSV:\n{csv_path}")