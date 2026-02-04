import torch
from dino_wrapper import DinoV3FeatureWrapper

IMGSZ = 512

dummy = torch.randn(1, 3, IMGSZ, IMGSZ)

wrapper = DinoV3FeatureWrapper()

wrapper.eval()
with torch.no_grad():
    out = wrapper(dummy)

torch.onnx.export(
    wrapper.eval(),
    dummy,
    f"vit_large_dinov3_features_{IMGSZ}.onnx",
    input_names=["input"],
    output_names=["cls", "patches"],
    opset_version=18,
    do_constant_folding=True,
    verify=False,
)
