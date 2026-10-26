#!/usr/bin/env bash
# Run on YOUR LAPTOP before the experiment ends. $HOME on CloudLab is shared NFS,
# so ~/logs on node0 already contains both node0/ and node1/ folders.
#   ./09_scp_results.sh <cloudlab_user>@<node0 hostname from CloudLab UI>
set -euo pipefail
HOST=${1:?user@node0.hostname}
DST=./part7_results_$(date +%Y%m%d_%H%M)
mkdir -p "$DST"
scp -r "$HOST:~/logs" "$DST/"
scp    "$HOST:~/.bashrc" "$DST/bashrc.txt"
scp -r "$HOST:~/gpunetio/install/examples" "$DST/examples_bin" 2>/dev/null || true
echo "saved to $DST"; find "$DST" -type f | sort
