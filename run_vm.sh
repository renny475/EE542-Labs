#!/usr/bin/env bash
# 在 AWS VM 上執行 gpu_acceleration.py (先跑過 setup_vm.sh)。參數會原樣傳下去，例如:
#   bash run_vm.sh --epochs 10
#
# DLAMI 的 /etc/profile.d/dlami.sh 把系統 CUDA 放進 LD_LIBRARY_PATH，會讓 pip 裝的 cuDNN
# 載到 /usr/local/cuda 裡另一個版本的 libcudnn_graph，conv 時報 CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED。
# PyTorch wheel 自帶完整的 CUDA 函式庫，所以只在這裡拿掉，不動系統設定 (其他 .cu 作業還要用)。
set -euo pipefail
cd "$(dirname "$0")"
exec env -u LD_LIBRARY_PATH /opt/dlami/nvme/lab7/.venv/bin/python gpu_acceleration.py "$@"
