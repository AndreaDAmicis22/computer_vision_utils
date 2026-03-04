import timm
import torch

MODEL_SIZE_MAP = {
    "h": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "l": "vit_large_patch16_dinov3.lvd1689m",
    "b": "vit_base_patch16_dinov3.lvd1689m",
    "sp": "vit_small_plus_patch16_dinov3.lvd1689m",
    "s": "vit_small_patch16_dinov3.lvd1689m",
}


class DinoV3FeatureWrapper(torch.nn.Module):
    def __init__(self, model_size="b"):
        super().__init__()
        self.model = timm.create_model(
            MODEL_SIZE_MAP[model_size],
            pretrained=True,
        )
        self.model.eval()

    def forward(self, x):
        tokens = self.model.forward_features(x)  # (B, 1+N, D)

        cls = tokens[:, 0, :]  # (B, D)
        patches = tokens[:, 1:, :]  # (B, N, D)

        return cls, patches
