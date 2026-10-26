#!/usr/bin/env bash
# Part 7 / Step 7.7 — build open-source GPUNetIO + examples for A30 (sm_80).
#
# !!! CloudLab mounts $HOME over NFS and SHARES it between node0 and node1.
# !!! So build ONCE (on node0). node1 only needs the env lines, which are in the shared ~/.bashrc.
# !!! Running this on both nodes at the same time will corrupt the build.
set -euo pipefail
LOG=$HOME/logs/$(hostname -s); mkdir -p "$LOG"
exec > >(tee -a "$LOG/7.7_build_gpunetio.log") 2>&1
echo "=== [$(date)] 7.7 build GPUNetIO on $(hostname) ==="

export PATH=/usr/local/cuda/bin:$PATH
nvcc --version | tail -1

[ -d "$HOME/gpunetio" ] || git clone https://github.com/NVIDIA-DOCA/gpunetio "$HOME/gpunetio"
cd "$HOME/gpunetio"
make -j"$(nproc)" install install_examples PREFIX="$HOME/gpunetio/install" CUDA_ARCH=80

echo "--- installed ---"
find "$HOME/gpunetio/install" -name '*.so*' -o -type f -perm -u+x | sort

# env (shared ~/.bashrc, so both nodes get it)
grep -q 'gpunetio/install/lib' "$HOME/.bashrc" || cat >> "$HOME/.bashrc" <<'EOF'
export LD_LIBRARY_PATH=$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:$LD_LIBRARY_PATH
export EX=$HOME/gpunetio/install/examples
EOF

# smoke test: can the examples find their libs?
export LD_LIBRARY_PATH=$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}
export EX=$HOME/gpunetio/install/examples
for e in gpunetio_verbs_write_lat gpunetio_verbs_put_bw gpunetio_verbs_write_bw; do
  echo "--- ldd $e (missing libs?)"; ldd "$EX/$e" | grep -i 'not found' || echo "  ok"
done
echo "=== done. On BOTH nodes: source ~/.bashrc  &&  ls \$EX ==="
