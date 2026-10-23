import torch, torchvision
model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
with torch.no_grad():
    for B in [1, 8, 32]:
        x = torch.randn(B, 3, 224, 224, device="cuda")
        for _ in range(10): model(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(100): model(x)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 100
        print(f"[pytorch fp32] batch={B:2d}  {ms:7.3f} ms/iter  {B/ms*1000:8.0f} img/s")
