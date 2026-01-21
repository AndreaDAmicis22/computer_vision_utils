from pathlib import Path

from ultralytics import YOLO

# -------------------------------------------
# 1️⃣ Define the model
# -------------------------------------------
model = YOLO("yolo11n-obb.pt")

# -------------------------------------------
# 2️⃣ Dataset root
# -------------------------------------------
DATASET_ROOT = Path("/workspace/LSF001/datasets/LSF001_YOLO_OBB/")

# -------------------------------------------
# 3️⃣ Start training
# -------------------------------------------
results = model.train(
    data=str(DATASET_ROOT / "data.yaml"),
    imgsz=1500,  # Image size
    device="0",  # GPU id or 'cpu'
    epochs=130,  # Number of training epochs
    batch=8,  # Batch size
    name="TRAIN_LSF_TEST_OBB_1500",
    save_period=10,  # Save every N epochs
    cache=True,  # Cache images for faster training
    max_det=1000,  # Max detections per image
    resume=False,  # Resume from last checkpoint
    # ----------------------------
    # Optimizer / LR / Weight decay
    # ----------------------------
    lr0=0.01,  # initial learning rate
    lrf=0.0001,  # final learning rate fraction (cosine decay)
    momentum=0.937,  # SGD momentum
    weight_decay=0.0005,  # L2 regularization
    optimizer="SGD",  # optimizer type
    workers=4,  # dataloader workers
    # ----------------------------
    # Data augmentations
    # ----------------------------
    augment=True,
    degrees=0.25,  # random rotation
    translate=0.2,  # random translation
    scale=0.3,  # random scale
    shear=0.0,  # random shear
    perspective=0.0005,  # perspective distortion
    flipud=0.5,  # vertical flip probability
    fliplr=0.5,  # horizontal flip probability
    mosaic=1,  # enable mosaic augmentation
    mixup=1,  # enable mixup augmentation
    copy_paste=1,  # enable copy-paste augmentation
    hsv_h=0.0,  # HSV hue augmentation
    hsv_s=0.2,  # HSV saturation augmentation
    hsv_v=0.1,  # HSV value augmentation
)

# -------------------------------------------
# 4️⃣ Export trained model
# -------------------------------------------
model.export(format="onnx", imgsz=1504, simplify=True, nms=True, opset=18, dynamic=True)

# Export to OpenVINO
model.export(format="openvino", imgsz=1504, optimize=True)
