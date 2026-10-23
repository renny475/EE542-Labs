import sys, time
import torch, torchvision
import tensorrt as trt

TRT_TO_TORCH = {trt.float32: torch.float32, trt.float16: torch.float16}

class TRTInfer:
    """Takes a CUDA tensor as input, returns a CUDA tensor. No host round-trip."""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(open(engine_path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.in_dtype  = TRT_TO_TORCH[self.engine.get_tensor_dtype("input")]
        self.out_dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype("logits")]
        self.stream = torch.cuda.Stream()

    def infer(self, inp):                       # inp: CUDA tensor, (B,3,224,224)
        assert inp.is_cuda and inp.is_contiguous()
        inp = inp.to(self.in_dtype)             # no-op if dtype already matches
        B = inp.shape[0]
        out = torch.empty((B, 1000), dtype=self.out_dtype, device="cuda")
        self.ctx.set_input_shape("input", tuple(inp.shape))
        self.ctx.set_tensor_address("input",  inp.data_ptr())
        self.ctx.set_tensor_address("logits", out.data_ptr())
        self.stream.wait_stream(torch.cuda.current_stream())   # input ready before TRT runs
        ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
        assert ok, "execute_async_v3 failed"
        torch.cuda.current_stream().wait_stream(self.stream)   # readers wait for TRT
        return out

    def infer_sync(self, inp):
        out = self.infer(inp)
        self.stream.synchronize()
        return out

if __name__ == "__main__":
    torch.manual_seed(0)
    ref = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
    x = torch.randn(8, 3, 224, 224, device="cuda")
    with torch.no_grad():
        ref_out = ref(x)

    for name in ["fp32", "fp16"]:
        trt_model = TRTInfer(f"resnet18_{name}.engine")
        out = trt_model.infer_sync(x).float()
        max_diff = (out - ref_out).abs().max().item()
        same_argmax = (out.argmax(1) == ref_out.argmax(1)).all().item()
        print(f"[{name}] in={trt_model.in_dtype} out={trt_model.out_dtype} "
              f"max|diff| vs PyTorch={max_diff:.2e} argmax match={same_argmax}")

        # benchmark: CUDA events, warmup 10, measure 100
        for B in [1, 8, 32]:
            xb = torch.randn(B, 3, 224, 224, device="cuda")
            for _ in range(10): trt_model.infer_sync(xb)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100): trt_model.infer(xb)
            end.record(); torch.cuda.synchronize()
            ms = start.elapsed_time(end) / 100
            print(f"  batch={B:2d}  {ms:7.3f} ms/iter  {B/ms*1000:8.0f} img/s")
