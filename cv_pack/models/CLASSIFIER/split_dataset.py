import os
import random
from glob import glob

import cv2

random.seed(42)

side = "down"  # "up" o "down"
good = f"/workspace/dataset_{side}/good/"
bad = f"/workspace/dataset_{side}/bad/"
flipped = f"/workspace/dataset_{side}/flipped/"
# flipped_bad = f"/workspace/dataset_{side}/flipped_bad/"

DATASET_ROOT = f"/workspace/dataset_{side}"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")

CLASS_DIRS = {
    "good": good,
    "bad": bad,
    "flipped": flipped,
    # "flipped_bad": flipped_bad,
}

SPLIT_RATIO = 0.80  # 80% train – 20% val
ROTATE_PROB = 0.5  # 50% di probabilità di duplicare e ruotare

print("=" * 80)  # noqa: T201
print(f"DATASET SPLIT STARTED | side={side}")  # noqa: T201
print(f"Split ratio: {SPLIT_RATIO}")  # noqa: T201
print(f"Rotate probability: {ROTATE_PROB}")  # noqa: T201
print("=" * 80)  # noqa: T201

# ---------------------------------------
# 1. Crea le directory
# ---------------------------------------
for split in ["train", "val"]:
    for cls in CLASS_DIRS:
        os.makedirs(os.path.join(DATASET_ROOT, split, cls), exist_ok=True)


# ---------------------------------------
# 2. Copia immagini divise per classe
# ---------------------------------------
for cls_name, src_dir in CLASS_DIRS.items():
    images = sorted(glob(os.path.join(src_dir, "*.*")))

    print("-" * 80)  # noqa: T201
    print(f"Processing class: {cls_name}")  # noqa: T201
    print(f"Source dir: {src_dir}")  # noqa: T201
    print(f"Found images: {len(images)}")  # noqa: T201

    if len(images) == 0:
        continue

    split_index = int(len(images) * SPLIT_RATIO)

    # Selezione casuale con random.sample
    train_imgs = random.sample(images, split_index)
    val_imgs = [img for img in images if img not in train_imgs]

    print(f"Train images: {len(train_imgs)}")  # noqa: T201
    print(f"Val images: {len(val_imgs)}")  # noqa: T201

    # ---- funzione di copia e rotazione ----
    def copy_and_rotate(img_list, target_dir, ROTATE_PROB=0.5, twice=False):
        for img_path in img_list:
            # ---- Leggi immagine originale ----
            img = cv2.imread(img_path)

            # ---- median blur sempre ----
            blurred_img = cv2.medianBlur(img, 7)  # kernel 3-11
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)

            # ---- salva immagine sfocata nella cartella target ----
            cv2.imwrite(os.path.join(target_dir, f"{name}{ext}"), blurred_img)

            # ---- augmentation casuale: rotazioni + flip ----
            if random.random() < ROTATE_PROB:
                # ---- prima rotazione ----
                angle = random.uniform(0, 360)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                rotated_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                # ---- median blur sulla versione ruotata ----
                blurred_rotated = cv2.medianBlur(rotated_img, 7)
                cv2.imwrite(os.path.join(target_dir, f"{name}_rot{int(angle)}{ext}"), blurred_rotated)

                # ---- seconda rotazione + flip se richiesto ----
                if twice and random.random() < 0.5:
                    angle2 = random.uniform(0, 360)
                    M2 = cv2.getRotationMatrix2D((w / 2, h / 2), angle2, 1.0)
                    rotated_img2 = cv2.warpAffine(
                        img, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                    )
                    blurred_rotated2 = cv2.medianBlur(rotated_img2, 7)
                    flipped_img = cv2.flip(blurred_rotated2, 1)  # flip orizzontale
                    cv2.imwrite(os.path.join(target_dir, f"{name}_rot{int(angle2)}_flip{ext}"), flipped_img)

    # ---- copia train ----
    print(f"Copying TRAIN images for class '{cls_name}'")  # noqa: T201
    copy_and_rotate(train_imgs, os.path.join(TRAIN_DIR, cls_name), ROTATE_PROB=ROTATE_PROB)
    # ---- copia val ----
    print(f"Copying VAL images for class '{cls_name}'")  # noqa: T201
    copy_and_rotate(val_imgs, os.path.join(VAL_DIR, cls_name), ROTATE_PROB=0.7, twice=True)


print("=" * 80)  # noqa: T201
print("DATASET SPLIT COMPLETED SUCCESSFULLY")  # noqa: T201
print("=" * 80)  # noqa: T201
