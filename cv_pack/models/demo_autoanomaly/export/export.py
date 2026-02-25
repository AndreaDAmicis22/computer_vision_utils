import torch
from dino_wrapper import DinoV3FeatureWrapper

MODEL_SIZE_MAP = {
    "h": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "l": "vit_large_patch16_dinov3.lvd1689m",
    "b": "vit_base_patch16_dinov3.lvd1689m",
    "sp": "vit_small_plus_patch16_dinov3.lvd1689m",
    "s": "vit_small_patch16_dinov3.lvd1689m",
}

IMGSZ = 464
SIZE = "b"

dummy = torch.randn(1, 3, IMGSZ, IMGSZ)

wrapper = DinoV3FeatureWrapper(SIZE)

wrapper.eval()
with torch.no_grad():
    out = wrapper(dummy)

torch.onnx.export(
    wrapper.eval(),
    dummy,
    f"../assets/{MODEL_SIZE_MAP[SIZE]}_{IMGSZ}.onnx",
    input_names=["input"],
    output_names=["cls", "patches"],
    opset_version=18,
    do_constant_folding=True,
    verify=False,
)
