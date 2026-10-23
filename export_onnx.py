import torch, torchvision

def export(model, x, path):
    torch.onnx.export(
        model, x, path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    print(f"wrote {path}")

model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()

# FP32
x32 = torch.randn(1, 3, 224, 224, device="cuda")
export(model, x32, "resnet18_fp32.onnx")

# FP16: 權重和輸入都轉成 half，ONNX graph 裡的 initializer 才會是 float16
model16 = model.half()
x16 = x32.half()
export(model16, x16, "resnet18_fp16.onnx")
