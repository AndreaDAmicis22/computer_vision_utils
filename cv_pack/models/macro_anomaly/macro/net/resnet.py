import timm
import torch
import torch.nn as nn


class ResNetFeatureExtractor(nn.Module):
    """Feature extractor that takes ResNet features at the specified layers."""

    def __init__(self, model_name: str, model_path: str, layers: list[str]):
        super().__init__()

        self.resnet = timm.create_model(model_name, checkpoint_path=model_path)
        self.layers = layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Performs a forward pass."""
        maps = []
        for name, module in list(self.resnet.named_children())[:-2]:
            x = module(x)
            if name in self.layers:
                maps.append(x)
        return maps
