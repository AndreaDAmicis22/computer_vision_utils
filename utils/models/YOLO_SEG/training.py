import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from sklearn.model_selection import train_test_split
from ultralytics import YOLO, settings


def write_stage_txt(path, items):
    with open(path, "w") as file:
        file.writelines(items)


def prepare_annotations(root_path: Path):
    # Copy annotations to local directory in order to be able to create symlinks
    tmp_path = tempfile.TemporaryDirectory()
    shutil.copytree(root_path, tmp_path.name, dirs_exist_ok=True)

    root_path = Path(tmp_path.name)

    # Link images folder to /mnt/nas
    images_path = root_path / "data" / "images" / "Train"
    images_path.mkdir(parents=True)
    symlink_path = images_path / "nas"
    os.symlink("/mnt/nas", symlink_path)

    shutil.move(root_path / "labels", root_path / "data" / "labels")

    original_train_file_path = root_path / "Train.txt"

    with open(original_train_file_path) as file:
        items = file.readlines()

    items = ["./" + x for x in items]

    train_file_path = root_path / "train.txt"
    val_file_path = root_path / "val.txt"
    test_file_path = root_path / "test.txt"

    seed = 42
    test_size = 0.1
    # 0.111... * 0.9 = 0.1
    val_size = test_size / (1 - test_size)

    x_train, x_test = train_test_split(items, test_size=test_size, random_state=seed)
    x_train, x_val = train_test_split(x_train, test_size=val_size, random_state=seed)

    write_stage_txt(train_file_path, x_train)
    write_stage_txt(val_file_path, x_val)
    write_stage_txt(test_file_path, x_test)

    original_config_file_path = root_path / "data.yaml"
    with open(original_config_file_path) as file:
        lines = file.readlines()

    lines = lines[1:-1]
    lines.append("train: train.txt\n")
    lines.append("val: val.txt\n")
    lines.append("test: test.txt\n")

    with open(original_config_file_path, "w") as file:
        file.writelines(lines)

    return tmp_path


def main():
    # Export dataset from CVAT as Ultralytics YOLO. Visit:
    # https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/
    root_path = Path("/mnt/nas/datasets/tnr004/ricciolo/20260121")

    with prepare_annotations(root_path) as tmp_path:
        tmp_root_path = Path(tmp_path)

        config_path = tmp_root_path / "data.yaml"
        project = "/workspace/src/TNR004/projects/tnr005/segmentation"

        # Even when logging to MLFlow, ultralytics still saves file locally
        os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"
        settings.update({"mlflow": True})

        model = YOLO("yolo11n-seg.pt")
        results = model.train(
            # Train settings
            data=config_path,
            imgsz=1504,  # Image size
            epochs=150,  # Number of training epochs
            batch=8,  # Batch size
            cache=False,
            device=-1,
            project=project,
            resume=False,  # Resume from last checkpoint
            # max_det=1000,  # Max detections per image
            # save_period=50,  # Save every N epochs
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
            degrees=180,  # random rotation
            translate=0.1,  # random translation
            scale=0.1,  # random scale
            shear=5,  # random shear
            perspective=0.0,  # perspective distortion
            flipud=0.5,  # vertical flip probability
            fliplr=0.5,  # horizontal flip probability
            mosaic=0.2,  # enable mosaic augmentation
            mixup=0.0,  # enable mixup augmentation
            copy_paste=0.0,  # enable copy-paste augmentation
            hsv_h=0.1,  # HSV hue augmentation
            hsv_s=0.1,  # HSV saturation augmentation
            hsv_v=0.1,  # HSV value augmentation
        )

        mlflow.log_param("data_original", str(root_path / "data.yaml"))
        mlflow.end_run()

        # Reset settings to default values
        settings.reset()

        model = YOLO(results.save_dir / "weights/best.pt")
        model.val(
            data=config_path,
            imgsz=1504,
            split="test",
            save_json=True,
            half=False,
            project=project,
        )

        model.export(format="onnx", imgsz=1504, simplify=True, nms=True, opset=18, dynamic=True)
        model.export(format="openvino", imgsz=1504, optimize=True)


if __name__ == "__main__":
    main()
