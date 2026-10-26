#!/usr/bin/env bash
# Part 7 / Steps 7.8–7.10 — one wrapper for every test, logs everything.
#
#   ./06_run_test.sh <test> <role>
#   test : lat_p1 | lat_p2 | lat_p0 | put_bw | write_bw | ib_lat | ib_bw
#   role : server (run on node0 FIRST)  |  client (run on node1 AFTER server is up)
#
# Fill in the 3 values below from 03_check_hw.sh output (after the IOMMU reboot!).
GPU_PCI="${GPU_PCI:-25:00.0}"        # e.g. 25:00.0
RDMA_DEV="${RDMA_DEV:-rocep65s0}"    # from `ibv_devices` AFTER the 7.6 reboot
SERVER_IP=10.10.1.1
GID=3                                # RoCE v2 over IPv4

set -uo pipefail
test=${1:?test}; role=${2:?server|client}
LOG=$HOME/logs/$(hostname -s); mkdir -p "$LOG"
out="$LOG/${test}_${role}.log"
export DOCA_GPUNETIO_LOG=6
export LD_LIBRARY_PATH=$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
EX=$HOME/gpunetio/install/examples

[ "$role" = client ] && C="-c $SERVER_IP" || C=""

case $test in
  lat_p1)   cmd="$EX/gpunetio_verbs_write_lat -g $GPU_PCI -d $RDMA_DEV -l $GID -p 1 $C" ;;
  lat_p2)   cmd="$EX/gpunetio_verbs_write_lat -g $GPU_PCI -d $RDMA_DEV -l $GID -p 2 $C" ;;   # expected: Can't export UAR to GPU
  lat_p0)   cmd="$EX/gpunetio_verbs_write_lat -g $GPU_PCI -d $RDMA_DEV -l $GID -p 0 $C" ;;   # expected: hangs → Ctrl+C after ~30s
  put_bw)   cmd="$EX/gpunetio_verbs_put_bw   -g $GPU_PCI -d $RDMA_DEV -l $GID -p 1 $C" ;;   # server: Ctrl+C after client finishes
  write_bw) cmd="$EX/gpunetio_verbs_write_bw -g $GPU_PCI -d $RDMA_DEV        -p 1 $C" ;;   # NO -l (rejected); uses GID 0 = link-local IPv6
  ib_lat)   cmd="ib_write_lat -d $RDMA_DEV -x $GID ${C:+$SERVER_IP}" ;;
  ib_bw)    cmd="ib_write_bw  -d $RDMA_DEV -x $GID --report_gbits ${C:+$SERVER_IP}" ;;
  *) echo "unknown test $test"; exit 1 ;;
esac

{
  echo "### $(date)  host=$(hostname)  test=$test role=$role"
  echo "### cmd: $cmd"
  echo "### cmdline: $(cat /proc/cmdline)"
  echo
  eval "$cmd"
  echo; echo "### exit=$?  $(date)"
} 2>&1 | tee "$out"
echo ">>> log: $out"
