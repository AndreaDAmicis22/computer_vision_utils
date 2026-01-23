import datetime
import shutil
from pathlib import Path

import torch
import torchvision.transforms as T
from ultralytics import YOLO, settings
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.split import autosplit
from ultralytics.models.yolo.classify import (
    ClassificationTrainer,
    ClassificationValidator,
)


class CustomizedDataset(ClassificationDataset):
    """A customized dataset class for image classification with enhanced data augmentation transforms."""

    def __init__(self, root: str, args, augment: bool = False, prefix: str = ""):
        """Initialize a customized classification dataset with enhanced data augmentation transforms."""
        super().__init__(root, args, augment, prefix)

        # Add your custom training transforms here
        train_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz), T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=args.fliplr),
                T.ColorJitter(
                    brightness=args.hsv_v,
                    contrast=args.hsv_v,
                    saturation=args.hsv_s,
                    hue=args.hsv_h,
                ),
                T.RandomAffine(
                    degrees=0,
                    scale=(args.scale, 1.0),
                    interpolation=T.InterpolationMode.BILINEAR,
                    fill=114,
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor(0),
                    std=torch.tensor(1),
                ),
            ]
        )

        # Add your custom validation transforms here
        val_transforms = T.Compose(
            [
                T.Resize((args.imgsz, args.imgsz), T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor(0),
                    std=torch.tensor(1),
                ),
            ]
        )
        self.torch_transforms = train_transforms if augment else val_transforms


class CustomizedTrainer(ClassificationTrainer):
    """A customized trainer class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build a customized dataset for classification training and the validation during training."""
        return CustomizedDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)


class CustomizedValidator(ClassificationValidator):
    """A customized validator class for YOLO classification models with enhanced dataset handling."""

    def build_dataset(self, img_path: str, mode: str = "train"):
        """Build a customized dataset for classification standalone validation."""
        return CustomizedDataset(
            root=img_path,
            args=self.args,
            augment=mode == "train",
            prefix=self.args.split,
        )


def split_dataset(images_dir: Path, dataset_dir: Path):
    if dataset_dir.exists():
        dataset_dir.rmdir()

    autosplit(
        path=images_dir,
        weights=(0.8, 0.1, 0.1),
    )

    split_file_prefix = "autosplit_"
    split_file_extension = "txt"
    stages = ["train", "val", "test"]
    for stage in stages:
        filename = split_file_prefix + stage + "." + split_file_extension
        split_file_path = images_dir.parent / filename

        stage_dir = dataset_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        with open(split_file_path) as split_file:
            lines = split_file.read().splitlines()
            for line in lines:
                img_path = images_dir.parent / line[2:]

                label_dir = stage_dir / img_path.parent.name
                label_dir.mkdir(exist_ok=True)

                new_img_path = label_dir / img_path.name
                shutil.copy(img_path, new_img_path)

        split_file_path.unlink()


def main():
    # Even when logging to MLFlow, ultralytics still saves file locally
    settings.update({"mlflow": True})

    # Export dataset from CVAT as Ultralytics YOLO. Visit:
    # https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/

    root_path = Path("/mnt/nas/datasets/tnr004/ricciolo/20260121_cls")
    dataset_dir = root_path / "dataset"

    generate_dataset = False
    if generate_dataset:
        images_dir = root_path / "images"
        split_dataset(images_dir, dataset_dir)

    project = "projects/tnr005/classification"
    name = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # model = YOLO("yolo11-cls-resnet18.yaml")
    model = YOLO("yolo11n-cls.pt")

    model.train(
        # Train settings
        data=dataset_dir,
        trainer=CustomizedTrainer,
        epochs=100,
        batch=8,
        imgsz=640,
        cache=False,
        device=-1,
        project=project,
        name=name,
        resume=False,
        # Augmentation
        hsv_h=0.25,
        hsv_s=0.5,
        hsv_v=0.4,
        scale=0.75,
    )

    # Reset settings to default values
    settings.reset()

    model.export(format="onnx", imgsz=640, simplify=True, opset=18, dynamic=True)


if __name__ == "__main__":
    main()
