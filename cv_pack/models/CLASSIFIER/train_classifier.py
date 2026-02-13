import os

import matplotlib.pyplot as plt
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# -----------------------------
# Configurazioni
# -----------------------------
side = "down"  # 'up' o 'down'
DATA_DIR = f"/workspace/dataset_{side}/7_median_filter"  # cartella con subfolder per classe
BATCH_SIZE = 32
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 80
LAMBDA_OVERFIT = 0.3  # peso per la penalizzazione dell'overfitting
LR = 1e-3
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(os.listdir(os.path.join(DATA_DIR, "train")))
# NAME_BACKBONE = "mobile_netv3_small"
# NAME_BACKBONE = "mobile_netv3_large"
NAME_BACKBONE = "efficientnet_b0"
OUTPUT_DIR = f"/workspace/src/LSF001/classifier_{side}_{NAME_BACKBONE}/"
NAME = f"classifier_{NAME_BACKBONE}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Trasformazioni
# -----------------------------
train_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
        transforms.RandomApply([transforms.RandomRotation(45)], p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.15)], p=0.5
        ),
        transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# -----------------------------
# Dataset e DataLoader
# -----------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# -----------------------------
# Modello
# -----------------------------
# weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
# model = models.mobilenet_v3_large(weights=weights)
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Recupera il numero di feature in ingresso all'ultimo classificatore
in_features = model.classifier[0].in_features

# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.3),
#     nn.Linear(in_features, 512),  # 1° FC
#     nn.ReLU(),
#     nn.Dropout(p=0.2),
#     nn.Linear(512, 256),  # 2° FC
#     nn.ReLU(),
#     nn.Dropout(p=0.2),
#     nn.Linear(256, NUM_CLASSES),  # 3° FC → logits
# )

model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 512),  # 1° FC
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(512, 256),  # 2° FC
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 128),  # 3° FC
    nn.ReLU(),
    nn.Dropout(p=0.15),
    nn.Linear(128, NUM_CLASSES),  # 4° FC → logits
)

model = model.to(DEVICE)

# -----------------------------
# Loss & Optimizer & Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

# -----------------------------
# Training Loop
# -----------------------------
best_score = -float("inf")
best_f1 = 0.0
early_stop_counter = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []
print("=" * 80)  # noqa: T201
print("Inizio training...🚀")  # noqa: T201
for _epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / total
    train_acc = correct / total
    train_f1 = f1_score(all_labels, all_preds, average="macro")
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)

    # Validation
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)

    diff_f1 = abs(train_f1 - val_f1)
    combined_score = val_f1 - LAMBDA_OVERFIT * diff_f1

    print(  # noqa: T201
        f"[Epoch {_epoch + 1:03d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train F1: {train_f1:.4f} || "
        f"Val Loss: {val_loss:.4f} | "
        f"Val F1: {val_f1:.4f} || "
        f"ΔF1: {diff_f1:.4f} | "
        f"Score: {combined_score:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
        f"Early Stop Counter: {early_stop_counter}"
    )

    # Aggiorna LR
    scheduler.step()
    if combined_score > best_score:
        best_epoch = _epoch
        best_score = combined_score
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_model_{side}.pth"))
        print(  # noqa: T201
            f"⭐ New BEST model at epoch {_epoch + 1} | Best score: {best_score:.4f}"
        )

        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print(  # noqa: T201
                f"⛔ STOP TRAINING at epoch {_epoch + 1} (best epoch: {best_epoch + 1})"
            )

            break

print("TRAINING COMPLETED 🚀")  # noqa: T201
print(f"Best epoch: {best_epoch + 1} ✨")  # noqa: T201
print(f"Best val F1-score: {best_f1:.4f} ✨")  # noqa: T201
print(f"Best combined score: {best_score:.4f} ✨")  # noqa: T201
print("=" * 80)  # noqa: T201

# -----------------------------
# Salvataggio grafici
# -----------------------------
# Loss
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Val Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.yticks(torch.arange(0.3, 1.05, 0.05))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

# F1-score
plt.figure(figsize=(12, 5))
plt.plot(train_f1s, label="Train F1")
plt.plot(val_f1s, label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.title("Train vs Val F1-score")
plt.grid(True, linestyle="--", alpha=0.6)
plt.yticks(torch.arange(0, 1.05, 0.05))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_curve.png"))
plt.close()


# -----------------------------
# Esportazione ONNX
# -----------------------------
print("\nEXPORTING ONNX MODEL...")  # noqa: T201
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, f"best_model_{side}.pth")))
model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
onnx_path = os.path.join(OUTPUT_DIR, f"best_model_{side}.onnx")
print(f"ONNX path: {onnx_path}")  # noqa: T201

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print("ONNX export completed successfully ✅")  # noqa: T201
print("=" * 80)  # noqa: T201


# -----------------------------
# Inferenza ONNX Runtime
# -----------------------------
def infer_onnx(image_tensor):
    ort_session = ort.InferenceSession(os.path.join(OUTPUT_DIR, f"best_model_{side}.onnx"))
    ort_inputs = {"input": image_tensor.cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = ort_outs[0]
    return preds.argmax(axis=1)


# Esempio di inferenza
sample_img, _ = val_dataset[0]
sample_img = sample_img.unsqueeze(0).to(DEVICE)
pred_class = infer_onnx(sample_img)
