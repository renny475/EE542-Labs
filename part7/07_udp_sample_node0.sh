#!/usr/bin/env bash
# Part 7 / Step 7.5 — node0 ONLY. Runs gpunetio_simple_receive inside the DOCA container.
# Expected on ConnectX-6 (non-Dx): stops at DOCA Flow port creation with "Failed to get hws cap".
# Pull the image EARLY (22 GB):  sudo docker pull $IMG   (started by 08_pull_doca.sh)
NIC_PCI="${NIC_PCI:-41:00.0}"   # PCI of the 10.10.1.1 interface (ethtool -i <iface> | grep bus-info)
GPU_PCI="${GPU_PCI:-25:00.0}"
IMG=nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host

LOG=$HOME/logs/$(hostname -s); mkdir -p "$LOG"
echo "### $(date) 7.5 simple_receive NIC=$NIC_PCI GPU=$GPU_PCI" | tee "$LOG/7.5_simple_receive.log"

sudo docker run --rm -i --privileged --net=host --gpus all \
  -v /dev/infiniband:/dev/infiniband \
  -v /dev/hugepages:/dev/hugepages \
  -v /opt/mellanox/gdrcopy/src:/opt/gdrcopy:ro \
  "$IMG" bash -c "
    set -x
    export LD_LIBRARY_PATH=/opt/gdrcopy:\$LD_LIBRARY_PATH
    cd /opt/mellanox/doca/samples/doca_gpunetio/gpunetio_simple_receive
    meson setup /tmp/build && ninja -C /tmp/build
    /tmp/build/doca_gpunetio_simple_receive -n $NIC_PCI -g $GPU_PCI -e 0
  " 2>&1 | tee -a "$LOG/7.5_simple_receive.log"
echo ">>> log: $LOG/7.5_simple_receive.log   (grep -i 'hws' it)"
