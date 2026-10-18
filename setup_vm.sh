#!/usr/bin/env bash
# 在 AWS GPU VM (DLAMI) 上建立 Lab7 的執行環境。
# venv 和資料集放在 instance store NVMe (/opt/dlami/nvme)：根目錄 EBS 空間不夠裝 CUDA 版 PyTorch。
# 注意: instance store 在 stop/start 後會被清空，重開機後重新執行本腳本即可 (約 3-5 分鐘)。
# 用法: 在程式碼所在目錄執行 bash setup_vm.sh
set -euo pipefail

SCRATCH=/opt/dlami/nvme/lab7
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
export TMPDIR="$SCRATCH/tmp"   # pip 解壓縮 wheel 也要大量暫存空間，不能用根目錄的 /tmp
mkdir -p "$TMPDIR" "$SCRATCH/data/data"

if [ ! -x "$SCRATCH/.venv/bin/python" ]; then
    python3 -m venv "$SCRATCH/.venv"
    "$SCRATCH/.venv/bin/pip" install --no-cache-dir -q --upgrade pip
    "$SCRATCH/.venv/bin/pip" install --no-cache-dir -q torch torchvision pyarrow pillow numpy
fi
env -u LD_LIBRARY_PATH "$SCRATCH/.venv/bin/python" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

cd "$SCRATCH/data/data"
for i in 00000 00001; do
    f="train-$i-of-00002.parquet"
    [ -f "$f" ] || curl -sSLO "https://huggingface.co/datasets/microsoft/cats_vs_dogs/resolve/main/data/$f"
done

cd "$SCRATCH"
[ -d data/train ] || "$SCRATCH/.venv/bin/python" "$CODE_DIR/prepare_data.py" data/data data

# 讓程式碼目錄下的 data/ 指向 NVMe 上的資料，gpu_acceleration.py 的預設路徑不用改
ln -sfn "$SCRATCH/data" "$CODE_DIR/data"

echo
echo "Ready. Run:"
echo "  bash $CODE_DIR/run_vm.sh --epochs 10"
