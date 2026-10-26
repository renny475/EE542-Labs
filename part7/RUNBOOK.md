# Part 7 Runbook — 4-hour slot (today 6 pm)

## Before 6 pm (no machines needed)

- [ ] `profile.py` → new **public** GitHub repo (alone) → CloudLab → DOCA-GPU project → Create Profile from repo → schedule experiment to start at slot start, Wisconsin cluster, team group.
- [ ] Experiment owner adds every teammate's SSH **public** key (CloudLab profile → Manage SSH Keys) **before** the experiment starts.
- [ ] Everyone has this folder locally. Roles: **A = node0**, **B = node1**, **C = log / clock / scp**.
- [ ] Copy scripts up as soon as nodes are reachable (`$HOME` is NFS-shared across both nodes, upload once):
  ```bash
  scp -r part7/ <user>@<node0-host>:~/part7 && ssh <user>@<node0-host> 'chmod +x ~/part7/*.sh'
  ```

## Facts to fill in during the slot

| key | node0 | node1 |
|---|---|---|
| IFACE (10.10.1.x) | | |
| NIC_PCI | | |
| GPU_PCI | | |
| RDMA_DEV **before** IOMMU reboot | | |
| RDMA_DEV **after** IOMMU reboot ← use this | | |
| topo (nvidia-smi topo -m) | | |
| firmware (ethtool -i) | | |

## Timeline

| t | node0 (A) | node1 (B) | notes |
|---|---|---|---|
| 0:00 | ssh in, `mountpoint /mydata` | same | if /mydata missing → profile bug, fix fast |
| 0:05 | `~/part7/01_setup_before_reboot.sh` | same, in parallel | ~15–25 min, ends with reboot #1 |
| ~0:30 | `~/part7/02_post_reboot.sh` | same | builds GDRCopy, sets hugepages/pm/ACS |
| ~0:35 | `~/part7/08_pull_doca.sh` (background, 22 GB) | — | |
| ~0:35 | `~/part7/03_check_hw.sh` | same | **Step 7.4** — fill table above |
| ~0:45 | wait for pull; `NIC_PCI=.. GPU_PCI=.. ~/part7/07_udp_sample_node0.sh` | — | **Step 7.5** — expect `Failed to get hws cap` |
| ~1:05 | `~/part7/04_iommu_off.sh` | same | **Step 7.6** reboot #2 |
| ~1:20 | `~/part7/02_post_reboot.sh` ; `cat /proc/cmdline` ; `ibv_devices` | same | **device name likely changed** → update table |
| ~1:25 | `~/part7/05_build_gpunetio.sh` (**node0 ONLY**, NFS-shared) | `source ~/.bashrc; ls $EX` after A finishes | **Step 7.7** |
| ~1:45 | `export GPU_PCI=.. RDMA_DEV=..` on both terminals | same | |
| | `~/part7/06_run_test.sh lat_p1 server` | then `~/part7/06_run_test.sh lat_p1 client` | **7.8** record latency per size |
| | `06_run_test.sh lat_p2 server` | `06_run_test.sh lat_p2 client` | expect `Can't export UAR to GPU`, cudaHostRegister 800 |
| | `06_run_test.sh lat_p0 server` | `06_run_test.sh lat_p0 client` | expect hang → Ctrl+C both after 30 s (optional, report bonus) |
| ~2:10 | `06_run_test.sh put_bw server` | `06_run_test.sh put_bw client` | **7.9** when client exits → Ctrl+C server → "Validation successful" |
| | `06_run_test.sh write_bw server` | `06_run_test.sh write_bw client` | no `-l`; same Ctrl+C dance |
| ~2:40 | `sudo apt-get install -y perftest` | same | **7.10** |
| | `06_run_test.sh ib_lat server` | `06_run_test.sh ib_lat client` | |
| | `06_run_test.sh ib_bw server` | `06_run_test.sh ib_bw client` | |
| ~3:00 | — | — | C: from laptop `./09_scp_results.sh <user>@<node0-host>` ; **open the files and check they are not empty** |
| ~3:15 | buffer: re-run anything that failed | | |
| 3:45 | **Terminate experiment** in CloudLab UI | | |

Rules: server (node0) **always starts first**. Every command already `tee`s to `~/logs/<node>/`. Don't `mlxconfig`, don't flash firmware.

## If something breaks

| symptom | fix |
|---|---|
| `nvidia-smi` fails after reboot #1 | `dmesg \| grep -i nvidia`; `sudo apt-get install --reinstall nvidia-open-580`; reboot again |
| `gdrdrv` won't build | `sudo apt-get install linux-headers-$(uname -r)` then `make driver` again |
| `libdoca_gpunetio_host.so` not found | `find ~/gpunetio/install -name 'libdoca*'` → add that dir to `LD_LIBRARY_PATH` |
| client: connection refused | server not up yet, or wrong `SERVER_IP`; `ping 10.10.1.1` from node1 |
| `ibv_devices` shows nothing after IOMMU reboot | `sudo modprobe mlx5_ib`; `dmesg \| grep mlx5` |
| docker pull out of space | `sudo docker info \| grep Root` must be `/mydata/docker`; if not, fix `/etc/docker/daemon.json`, `systemctl restart docker` |
| ACS script picked the BlueField instead of the CX-6 | `lspci -D \| grep Mellanox`, set `NIC_BDF` manually in `02_post_reboot.sh` to the bus-info from `ethtool -i <iface>` |
| 7.5 fails for a reason other than hws cap | still record it; the report needs the real error |
