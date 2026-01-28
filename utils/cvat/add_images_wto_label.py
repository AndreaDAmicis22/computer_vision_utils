import random
from pathlib import Path

# --- CONFIGURAZIONE ---

source_dir = Path("/mnt/nas/TNR004/ricciolo/20251126/Ricciolo/analisi13_14-10-25/Riccioli_13_10_2025/1_MAN")

labels_dir = Path(
    "~/Downloads/merge_total/labels/Train/nas/TNR004/ricciolo/20251126/Ricciolo/analisi13_14-10-25/Riccioli_13_10_2025/1_MAN"
).expanduser()
labels_dir.mkdir(parents=True, exist_ok=True)

train_txt_file = Path("~/Downloads/merge_total/Train.txt").expanduser()
train_txt_file.touch(exist_ok=True)

num_samples = 150
train_prefix = "data/images/Train"

# --- STEP 1: Lista immagini ---
all_images = list(source_dir.glob("*.jpg"))
if len(all_images) < num_samples:
    msg = f"Ci sono solo {len(all_images)} immagini, meno di {num_samples}"
    raise ValueError(msg)

# --- STEP 2: Random sample ---
sampled_images = random.sample(all_images, num_samples)

# --- STEP 3: Crea label vuote ---
for img_path in sampled_images:
    (labels_dir / f"{img_path.stem}.txt").touch()

# --- STEP 4: Append corretto a Train.txt ---
with open(train_txt_file, "a") as f:
    for img_path in sampled_images:
        relative_path = (
            Path(train_prefix)
            / "nas/TNR004/ricciolo/20251126/Ricciolo/analisi13_14-10-25/Riccioli_13_10_2025/1_MAN"
            / img_path.name
        )
        f.write(str(relative_path) + "\n")
