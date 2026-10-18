import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mlp import MLP2, batch_iter

BATCH_SIZE = 32
IMG_SIZE = 64
MLP_HIDDEN = 128
MLP_LR = 1e-2   # 和 mlp.py 的 train() 預設相同，NumPy 與 PyTorch MLP 用同一組超參數


def pick_device():
    # NVIDIA GPU 用 cuda，Apple Silicon (M1/M2/...) 的 GPU 用 mps，都沒有才退回 cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    # GPU 是非同步執行的，計時前要等它真的算完，量到的時間才準
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class CatDogCNN(nn.Module):
    """
    conv1: 3x64x64  -> 16x64x64 -> pool -> 16x32x32
    conv2: 16x32x32 -> 32x32x32 -> pool -> 32x16x16
    fc1:   32*16*16 -> 128
    fc2:   128      -> num_classes (ReLU, 配合 MSELoss 去逼近 one-hot)
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)    # padding=1 保持 HxW 不變
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 64 -> 32
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 32 -> 16
        x = torch.flatten(x, 1)                       # [B, 32*16*16]
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))   # 依作業規定輸出層也加 ReLU，搭配 MSELoss


class TorchMLP2(nn.Module):
    """和 mlp.py 的 MLP2 同架構 (Linear -> ReLU -> Linear)，用來做 NumPy vs PyTorch 的同模型比較"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def cnn_setup(device, num_classes):
    model = CatDogCNN(num_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # MSELoss 要求 output 和 target 同形狀: labels [B] 要先轉成 one-hot [B, num_classes]
    make_target = lambda labels: F.one_hot(labels, num_classes).float()
    return model, criterion, optimizer, make_target


def mlp_setup(device, num_classes):
    # 和 NumPy MLP2(mode='classification') 相同: softmax cross-entropy + 純 SGD
    model = TorchMLP2(3 * IMG_SIZE * IMG_SIZE, MLP_HIDDEN, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=MLP_LR)
    return model, criterion, optimizer, lambda labels: labels


def TrainingLoop(model, loader, criterion, optimizer, device, make_target, epochs, verbose):
    model.train()
    sync(device)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()                            # 清掉上一個 batch 累積的梯度
            outputs = model(images)                          # forward
            loss = criterion(outputs, make_target(labels))
            loss.backward()                                  # backprop
            optimizer.step()                                 # 更新權重

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)

        if verbose:
            print(f"  epoch {epoch:3d}/{epochs}  loss={total_loss / seen:.4f}  "
                  f"train_acc={correct / seen:.3f}")

    sync(device)
    return time.perf_counter() - start


def evaluate(model, loader, device):
    """回傳 (accuracy, 秒數)。時間包含把 batch 搬到 device，這是實際推論時也要付的成本"""
    model.eval()
    correct, seen = 0, 0
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():                 # 推論不需要梯度，省記憶體也更快
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
    sync(device)
    return correct / seen, time.perf_counter() - start


def load_tensors(subset):
    """先把圖片一次解碼成 tensor。之後所有版本都吃同一份記憶體中的資料，
    計時只量模型運算，不會被 JPEG 解碼 (純 CPU 工作) 稀釋掉 GPU 的差距"""
    # 不開 num_workers: macOS 用 spawn 啟動 worker，每個都要重新 import torch，實測反而慢 50 倍
    loader = DataLoader(subset, batch_size=256)
    xs, ys = zip(*loader)
    return torch.cat(xs), torch.cat(ys)


class TensorBatches:
    """像 DataLoader 一樣每次 iterate 走完一個 epoch，但資料已經整批在記憶體裡，直接切 tensor。
    DataLoader(TensorDataset) 會逐張取出再 collate，實測 1000 張的 block 要 0.17s，
    比 MLP 在 GPU 上的運算還久，會把 CPU/GPU 的差距稀釋掉"""
    def __init__(self, X, y, batch_size, shuffle=False):
        self.X, self.y, self.batch_size, self.shuffle = X, y, batch_size, shuffle

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n) if self.shuffle else None   # 每個 epoch 重新打亂
        for i in range(0, n, self.batch_size):
            if order is None:
                yield self.X[i:i + self.batch_size], self.y[i:i + self.batch_size]
            else:
                j = order[i:i + self.batch_size]
                yield self.X[j], self.y[j]


def run_torch(name, setup, device, train_data, test_data, num_classes, args):
    torch.manual_seed(0)   # 每個版本從同樣的初始權重和 shuffle 順序開始
    model, criterion, optimizer, make_target = setup(device, num_classes)
    train_loader = TensorBatches(*train_data, BATCH_SIZE, shuffle=True)
    test_loader = TensorBatches(*test_data, args.eval_block)

    # Warmup: GPU 第一次執行某個運算時要編譯 kernel、配置記憶體，不該算進計時。
    # 只做 forward/backward 不做 optimizer.step()，權重不會被改動
    model(train_data[0][:BATCH_SIZE].to(device)).sum().backward()
    optimizer.zero_grad()
    with torch.no_grad():
        model(test_data[0][:args.eval_block].to(device))
    sync(device)

    print(f"\n[{name} on {device}]")
    train_time = TrainingLoop(model, train_loader, criterion, optimizer, device,
                              make_target, args.epochs, verbose=True)
    acc, test_time = evaluate(model, test_loader, device)
    return dict(name=name, device=str(device), train=train_time, test=test_time, acc=acc)


def run_numpy(train_data, test_data, num_classes, args):
    # .numpy() 和 tensor 共用記憶體，reshape 也只是 view，不會多複製一份資料
    Xtr = train_data[0].reshape(len(train_data[0]), -1).numpy()
    ytr = train_data[1].numpy()
    Xte = test_data[0].reshape(len(test_data[0]), -1).numpy()
    yte = test_data[1].numpy()

    model = MLP2(Xtr.shape[1], MLP_HIDDEN, num_classes, mode="classification", seed=0)

    print("\n[NumPy MLP2 on cpu]")
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        total_loss, correct = 0.0, 0
        for Xb, yb in batch_iter(Xtr, ytr, BATCH_SIZE, shuffle=True, seed=epoch):
            probs = model.forward(Xb)
            total_loss += model.loss(yb) * len(yb)
            correct += (probs.argmax(axis=1) == yb).sum()
            model.step(model.backward(yb), lr=MLP_LR)
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={total_loss / len(ytr):.4f}  "
              f"train_acc={correct / len(ytr):.3f}")
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    correct = 0
    for i in range(0, len(yte), args.eval_block):
        correct += (model.forward(Xte[i:i + args.eval_block]).argmax(axis=1)
                    == yte[i:i + args.eval_block]).sum()
    test_time = time.perf_counter() - start
    return dict(name="NumPy MLP2", device="cpu", train=train_time, test=test_time,
                acc=correct / len(yte))


def print_results(results, n_test):
    print(f"\n{'model':<14}{'device':<8}{'train (s)':>11}{'test (s)':>11}"
          f"{'test img/s':>12}{'test acc':>10}")
    for r in results:
        print(f"{r['name']:<14}{r['device']:<8}{r['train']:>11.2f}{r['test']:>11.3f}"
              f"{n_test / r['test']:>12.0f}{r['acc']:>10.3f}")

    find = {(r["name"], r["device"]): r for r in results}
    gpu = next((r["device"] for r in results if r["device"] != "cpu"), None)
    if gpu is None:
        return
    print("\nSpeedup (baseline time / GPU time):")
    for base_key, fast_key in [(("NumPy MLP2", "cpu"), ("Torch MLP2", gpu)),
                               (("Torch MLP2", "cpu"), ("Torch MLP2", gpu)),
                               (("CatDogCNN", "cpu"), ("CatDogCNN", gpu))]:
        base, fast = find[base_key], find[fast_key]
        print(f"  {' '.join(base_key):<16} -> {' '.join(fast_key):<16}"
              f"train x{base['train'] / fast['train']:5.1f}   test x{base['test'] / fast['test']:5.1f}")


def main():
    parser = argparse.ArgumentParser(description="EE542 Lab7 Part III: GPU acceleration")
    parser.add_argument("data_dir", nargs="?", default="data/train",
                        help="ImageFolder 根目錄: <data_dir>/<class_name>/<image> (先跑 prepare_data.py)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--eval-block", type=int, default=1000,
                        help="測試時每一大塊 (block) 的圖片數")
    args = parser.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),   # 統一成 64x64，才能疊成同一個 batch
        transforms.ToTensor(),                     # PIL [0,255] HxWxC -> float [0,1] CxHxW
    ])
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    num_classes = len(dataset.classes)
    print(f"Classes: {dataset.class_to_idx}")
    print(f"Total images: {len(dataset)}")

    # Try: 取一個 batch 確認載入正確
    images, labels = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)))
    print(f"Batch images shape: {images.shape}")   # [B, 3, 64, 64]
    print(f"Batch labels shape: {labels.shape}")   # [B]

    # 自動切出訓練集和模型沒看過的測試集 (固定 seed，每次切法相同)
    n_test = int(len(dataset) * args.test_ratio)
    train_set, test_set = random_split(dataset, [len(dataset) - n_test, n_test],
                                       generator=torch.Generator().manual_seed(0))
    print(f"Train: {len(train_set)}  Test (unseen): {len(test_set)}")

    start = time.perf_counter()
    train_data = load_tensors(train_set)
    test_data = load_tensors(test_set)
    print(f"Decoded images in {time.perf_counter() - start:.1f}s")

    cpu = torch.device("cpu")
    results = [run_numpy(train_data, test_data, num_classes, args),
               run_torch("Torch MLP2", mlp_setup, cpu, train_data, test_data, num_classes, args)]
    if device != cpu:
        results.append(run_torch("Torch MLP2", mlp_setup, device, train_data, test_data,
                                 num_classes, args))
    results.append(run_torch("CatDogCNN", cnn_setup, cpu, train_data, test_data, num_classes, args))
    if device != cpu:
        results.append(run_torch("CatDogCNN", cnn_setup, device, train_data, test_data,
                                 num_classes, args))
    print_results(results, n_test)


if __name__ == "__main__":
    main()
