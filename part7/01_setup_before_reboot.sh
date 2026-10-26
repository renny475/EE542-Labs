#!/usr/bin/env bash
# Part 7 / Step 7.3 — run on BOTH nodes, right after the experiment is up.
# Installs: apt deps, NVIDIA open driver 580 + CUDA 13.0, Docker + NVIDIA
# Container Toolkit (data-root on /mydata), GDRCopy v2.5.2 source.
# Ends with a reboot (driver needs it).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
LOG=$HOME/logs/$(hostname -s)
mkdir -p "$LOG"
exec > >(tee -a "$LOG/01_setup.log") 2>&1

echo "=== [$(date)] 01_setup start on $(hostname) ==="

# ---------- 0. sanity ----------
mountpoint -q /mydata || { echo "ERROR: /mydata not mounted — check profile.py Blockstore"; exit 1; }
sudo chown "$USER":"$(id -gn)" /mydata

# ---------- 1. apt deps ----------
sudo apt-get update
sudo apt-get install -y build-essential git linux-headers-"$(uname -r)" \
    rdma-core ibverbs-utils libibverbs-dev infiniband-diags pciutils ethtool \
    check libsubunit-dev pkg-config jq curl ca-certificates gnupg

# ---------- 2. NVIDIA open driver 580 + CUDA 13.0 ----------
# https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html
cd /tmp
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y nvidia-open-580        # open kernel module (needed for dmabuf)
sudo apt-get install -y cuda-toolkit-13-0
# make sure the in-tree nouveau never grabs the GPU
echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u

# ---------- 3. Docker CE + NVIDIA Container Toolkit, data-root on /mydata ----------
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io nvidia-container-toolkit

sudo mkdir -p /mydata/docker
sudo nvidia-ctk runtime configure --runtime=docker          # writes /etc/docker/daemon.json
sudo jq '. + {"data-root": "/mydata/docker"}' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.new > /dev/null
sudo mv /etc/docker/daemon.json.new /etc/docker/daemon.json
sudo systemctl enable docker
sudo usermod -aG docker "$USER"

# ---------- 4. GDRCopy v2.5.2 source (build happens after reboot — needs nvidia driver) ----------
sudo mkdir -p /opt/mellanox
sudo chown "$USER":"$(id -gn)" /opt/mellanox
git clone -b v2.5.2 --depth 1 https://github.com/NVIDIA/gdrcopy /opt/mellanox/gdrcopy

# ---------- 5. env for later steps ----------
cat >> "$HOME/.bashrc" <<'EOF'

# ---- Part 7 ----
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}
EOF

echo "=== [$(date)] 01_setup done — REBOOTING in 5s (driver). After reboot run 02_post_reboot.sh ==="
sleep 5
sudo reboot
