import csv
import numpy as np
import torch

from infer_trt import TRTInfer

FILES = [
    "frames_rx.bin",
    "frames_rx_shuffled.bin",
    "frames_rx_duplicates.bin",
]

torch.set_grad_enabled(False)

# Load original test images.
original = np.load("images.npy")
assert original.shape == (16, 224, 224, 3)
assert original.dtype == np.uint8

# Load the existing FP16 TensorRT engine.
model = TRTInfer("resnet18_fp16.engine")
print("TensorRT input dtype:", model.in_dtype)
print("TensorRT output dtype:", model.out_dtype)

# Standard ImageNet normalization.
mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device="cuda",
    dtype=torch.float32
).view(1, 3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device="cuda",
    dtype=torch.float32
).view(1, 3, 1, 1)


def classify(images):
    """Transfer pinned CPU images and preprocess on GPU."""

    # Allocate page-locked (pinned) CPU memory.
    host = torch.empty(
        images.shape,
        dtype=torch.uint8,
        pin_memory=True
    )
    host.copy_(torch.from_numpy(images.copy()))

    # Warm up GPU transfer.
    gpu = torch.empty_like(host, device="cuda")
    gpu.copy_(host, non_blocking=True)
    torch.cuda.synchronize()

    # Measure CPU-to-GPU transfer with CUDA events.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(100):
        start.record()
        gpu.copy_(host, non_blocking=True)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    copy_ms = float(np.mean(times))

    # Preprocess entirely on GPU.
    x = gpu.permute(0, 3, 1, 2)
    x = x.float().div(255.0)
    x = (x - mean) / std
    x = x.to(model.in_dtype).contiguous()

    # Run FP16 TensorRT inference.
    logits = model.infer_sync(x)
    predictions = logits.argmax(dim=1).cpu().numpy()

    return predictions, copy_ms


print("\nClassifying original images...")
original_predictions, original_copy_ms = classify(original)

print("Original predictions:", original_predictions.tolist())

rows = []

for filename in FILES:
    print(f"\nTesting {filename}")

    received = np.fromfile(filename, dtype=np.uint8)
    assert received.size == original.size, (
        f"Unexpected file size: {filename}"
    )
    received = received.reshape(original.shape)

    # Verify exact image reconstruction.
    byte_match = np.array([
        np.array_equal(original[i], received[i])
        for i in range(len(original))
    ])

    predictions, copy_ms = classify(received)

    class_match = predictions == original_predictions

    for i in range(len(original)):
        print(
            f"Frame {i:02d}: "
            f"original={original_predictions[i]:4d} "
            f"received={predictions[i]:4d} "
            f"byte_match={bool(byte_match[i])} "
            f"class_match={bool(class_match[i])}"
        )

        rows.append({
            "file": filename,
            "frame": i,
            "original_class": int(original_predictions[i]),
            "received_class": int(predictions[i]),
            "byte_match": bool(byte_match[i]),
            "class_match": bool(class_match[i]),
            "copy_ms_batch": copy_ms,
        })

    print(
        f"Results: {int(class_match.sum())}/16 "
        f"classifications matched"
    )
    print(
        f"Byte-identical frames: {int(byte_match.sum())}/16"
    )
    print(f"Pinned CPU-to-GPU copy: {copy_ms:.4f} ms/batch")
    print(f"Average per image: {copy_ms / 16:.4f} ms")


with open("part5_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nResults saved to part5_results.csv")
