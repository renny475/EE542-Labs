#!/usr/bin/env bash
# Part 7 — run on BOTH nodes after EVERY reboot (reboot #1 after driver, reboot #2 after IOMMU).
# Restores what a reboot wipes: hugepages, gdrdrv module, GPU persistence mode, ACS off.
# First run also builds GDRCopy (needs the nvidia driver, so it couldn't be built in 01_).
set -euo pipefail
LOG=$HOME/logs/$(hostname -s)
mkdir -p "$LOG"
exec > >(tee -a "$LOG/02_post_reboot_$(date +%H%M%S).log") 2>&1
echo "=== [$(date)] 02_post_reboot on $(hostname) ==="

# ---------- 0. driver alive? ----------
nvidia-smi || { echo "ERROR: nvidia-smi failed — driver not loaded. Check: dmesg | grep -i nvidia"; exit 1; }
cat /proc/driver/nvidia/version
lsmod | grep -q '^nvidia ' && echo "nvidia module loaded"

# ---------- 1. GDRCopy build (idempotent) + load gdrdrv ----------
cd /opt/mellanox/gdrcopy
if [ ! -f src/libgdrapi.so ]; then
    make -j"$(nproc)" lib driver
fi
sudo ./insmod.sh
lsmod | grep gdrdrv || { echo "ERROR: gdrdrv not loaded"; exit 1; }

# ---------- 2. hugepages: 2048 x 2MB ----------
echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
mountpoint -q /dev/hugepages || sudo mount -t hugetlbfs nodev /dev/hugepages
grep -E 'HugePages_(Total|Free)' /proc/meminfo

# ---------- 3. GPU persistence mode ----------
sudo nvidia-smi -pm 1

# ---------- 4. ACS off on every PCIe bridge above the GPU and the NIC ----------
# Guide: setpci -s <bridge> ECAP_ACS+0x6.w=0000 ; verify ACSCtl all "-".
# We walk up from each endpoint to the root and clear ACSCtl on every bridge that has the ACS cap.
GPU_BDF=$(lspci -D | awk '/NVIDIA.*(3D|VGA)/{print $1; exit}')
NIC_BDF=$(lspci -D | awk '/Mellanox/{print $1; exit}')   # first Mellanox fn; refine if BlueField also present
echo "GPU=$GPU_BDF  NIC=$NIC_BDF"

disable_acs_chain () {
    local dev=$1
    local path
    path=$(readlink -f /sys/bus/pci/devices/"$dev")
    # every ancestor directory named like a BDF is a bridge
    for b in $(echo "$path" | tr '/' '\n' | grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$' | grep -v "^$dev$"); do
        if sudo lspci -s "$b" -vvv 2>/dev/null | grep -q ACSCtl; then
            sudo setpci -s "$b" ECAP_ACS+0x6.w=0000
            echo -n "$b  "; sudo lspci -s "$b" -vvv | grep ACSCtl
        fi
    done
}
disable_acs_chain "$GPU_BDF"
disable_acs_chain "$NIC_BDF"

# ---------- 5. report ----------
echo "--- /proc/cmdline ---"; cat /proc/cmdline
echo "--- ibv_devices ---";  ibv_devices
echo "--- topo ---";         nvidia-smi topo -m
echo "=== [$(date)] 02_post_reboot done ==="
