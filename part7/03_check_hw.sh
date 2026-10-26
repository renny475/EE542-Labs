#!/usr/bin/env bash
# Part 7 / Step 7.4 — run on BOTH nodes. Captures everything the report needs.
set -uo pipefail
LOG=$HOME/logs/$(hostname -s)
mkdir -p "$LOG"
OUT="$LOG/7.4_hardware.log"
{
echo "=== Step 7.4 hardware check on $(hostname) — $(date) ==="
echo; echo "## ip -br addr";            ip -br addr
IFACE=$(ip -br addr | awk '/10\.10\.1\./{print $1; exit}')
echo; echo "## experiment iface = $IFACE"
echo; echo "## ethtool -i $IFACE";      ethtool -i "$IFACE"
echo; echo "## ibv_devices";            ibv_devices
echo; echo "## ibv_devinfo (short)";    ibv_devinfo | grep -E 'hca_id|fw_ver|link_layer|state|port:'
echo; echo "## lspci | grep -i mellanox"; lspci | grep -i mellanox
echo; echo "## lspci | grep -i nvidia";   lspci | grep -i nvidia
echo; echo "## uname -r";               uname -r
echo; echo "## nvidia-smi";             nvidia-smi
echo; echo "## nvidia-smi topo -m";     nvidia-smi topo -m
echo; echo "## lspci -tv (bridge path)"; lspci -tv
echo; echo "## GID table (for -l 3 / -x 3)"
for d in /sys/class/infiniband/*; do
  echo "$(basename "$d"):"
  for g in "$d"/ports/1/gids/{0,1,2,3}; do
    [ -f "$g" ] && printf '  gid[%s] = %s  type=%s\n' "$(basename "$g")" "$(cat "$g")" "$(cat "$d/ports/1/gid_attrs/types/$(basename "$g")" 2>/dev/null)"
  done
done
echo; echo "## BAR1";                  nvidia-smi -q | grep -A3 'BAR1'
echo; echo "## hugepages";             grep -E 'HugePages_(Total|Free)' /proc/meminfo
echo; echo "## iommu cmdline";         cat /proc/cmdline
} 2>&1 | tee "$OUT"
echo
echo ">>> saved to $OUT"
echo ">>> Fill these into RUNBOOK.md:  IFACE=$IFACE  NIC_PCI=$(ethtool -i "$IFACE" | awk '/bus-info/{print $2}' | sed 's/^0000://')  GPU_PCI=$(lspci | awk '/NVIDIA.*(3D|VGA)/{print $1; exit}')  RDMA_DEV=$(ibv_devices | awk 'NR>2{print $1; exit}')"
