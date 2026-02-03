import torch

from dino_wrapper import DinoV3FeatureWrapper

dummy = torch.randn(1, 3, 512, 512)

wrapper = DinoV3FeatureWrapper()

wrapper.eval()
with torch.no_grad():
    out = wrapper(dummy)

torch.onnx.export(
    wrapper.eval(),
    dummy,
    "vit_large_dinov3_features.onnx",
    input_names=["input"],
    output_names=["cls", "patches"],
    opset_version=18,
    do_constant_folding=True,
    verify=False,
)

# torch.onnx.export(
#     wrapper.eval(),
#     dummy,
#     "vit_large_dinov3_features.onnx",
#     input_names=["input"],
#     output_names=["tokens"],
#     opset_version=18,
#     do_constant_folding=True,
#     verify=False,
# )
