import timm
import torch


class DinoV3FeatureWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_large_patch16_dinov3.lvd1689m",
            pretrained=True,
        )
        self.model.eval()

    def forward(self, x):
        tokens = self.model.forward_features(x)  # (B, 1+N, D)

        cls = tokens[:, 0, :]  # (B, D)
        patches = tokens[:, 1:, :]  # (B, N, D)

        return cls, patches
