# Part 7 report / video-log notes (pre-researched; verify against your logs)

Sources: DOCA 3.5.0 GPUNetIO Installation and Setup
(https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-installation-and-setup) and
NVIDIA-DOCA/gpunetio README (https://github.com/NVIDIA-DOCA/gpunetio).

## 7.4 — hardware vs. the guide

- Guide: NIC should be **ConnectX-6 Dx or later**, firmware compatible with the DOCA release.
- Guide: GPU↔NIC topology should be **PIX or PXB** (dedicated PCIe path); PHB acceptable if same host bridge + NUMA; **avoid NODE and SYS**.
- Our d7525: ConnectX-6 (not Dx), fw 20.41, topology **NODE** → GPUDirect traffic between NIC and GPU crosses the link between two PCIe host bridges inside one NUMA node (AMD EPYC: the two devices hang off different root complexes / IO dies). It still works, but peer-to-peer bandwidth is capped below a PIX path.
- Compare with the Part 6 cloud instance (fill in): NIC model, topology string, whether UDP sample ran.

## 7.5 — why `gpunetio_simple_receive` dies with `Failed to get hws cap`

- The Ethernet samples steer packets to a GPU-owned receive queue with **DOCA Flow in hardware-steering (HWS) mode**. On port start DOCA Flow queries the NIC's HWS capabilities (`hws cap`).
- HWS is implemented in ConnectX-6 **Dx** / ConnectX-7 / BlueField-2+ firmware. Plain ConnectX-6 firmware has no HWS → the capability query fails → DOCA Flow port creation aborts before any queue is created.
- This is consistent with the guide's "ConnectX-6 Dx or later" recommendation. The open-source GPUNetIO verbs path does not use DOCA Flow at all (it uses `ibverbs` + mlx5 DevX), which is why the RDMA tests in 7.8–7.10 work on the same card.
- Report should state: what the sample needs (DOCA Flow HWS), what the card provides (no HWS), and which cards the guide recommends.

## 7.8 — `-p 2` fails: `Can't export UAR to GPU`, `cudaHostRegister` error 800

- A **UAR** (User Access Region) is a page of the NIC's PCIe BAR holding the doorbell register for a QP. Ringing the doorbell = writing the WQE index to that MMIO address.
- `-p 2` (GPU SM DB) asks the CUDA kernel to write that register itself. For that the UAR page must be mapped into the GPU's address space: GPUNetIO calls `cudaHostRegister(uar_ptr, size, cudaHostRegisterIoMemory)` so the GPU can issue PCIe writes to the NIC BAR (peer-to-peer MMIO).
- Error 800 = `cudaErrorNotSupported`: the driver/platform refused to register the IO-memory page. Reasons on this box: GPU and NIC are on **different PCIe host bridges (NODE)** so P2P MMIO writes must traverse the inter-root-complex link, which AMD platforms/nvidia driver may not allow for peer MMIO; ACS/IOMMU state at that moment; `PeerMappingOverride` not set.
- What the GPU would need to ring the doorbell: (1) a PCIe path where the GPU can do peer-to-peer writes to the NIC BAR (PIX/PXB, same root complex, ACS off), (2) the NVIDIA driver allowing `cudaHostRegisterIoMemory` on that BAR, (3) NIC firmware exposing the UAR as mappable (BlueFlame/DB page). Without it, the fallback is **CPU proxy**.
- `-p 0` (AUTO): library detects the UAR export failure and logs "Enabling CPU proxy mode", but the *example programs* only spawn the proxy thread when you pass `-p 1` → nobody calls `doca_gpu_verbs_cpu_proxy_progress()` → kernel spins forever. Design gap in the examples, not a hardware fault.

## 7.10 — the table and the two columns

| | GPUNetIO `-p 1` | perftest |
|---|---|---|
| where the data lives | GPU HBM (registered via dmabuf / ibv_reg_dmabuf_mr) | host DRAM |
| who builds the WQE | CUDA kernel, in GPU memory | CPU, in host memory |
| who rings the doorbell | **CPU proxy thread** (GPU writes a flag, CPU polls, CPU writes UAR) | CPU |
| who polls the CQE | CUDA kernel | CPU |
| NIC data path | NIC ↔ GPU BAR1 over PCIe P2P (crosses host-bridge link here) | NIC ↔ DRAM |

Expected shape of results (check against the logs):
- **Small messages**: GPUNetIO latency ≥ perftest. Extra hop: GPU flag → CPU proxy poll → doorbell. Plus GPU-side WQE build/CQ poll are slower than CPU for 1 op.
- **64 KB bandwidth**: both approach link rate (~100 Gb/s class); if GPUNetIO is lower, the NODE P2P path is the suspect. Compute: `64KB × msg_rate` and compare to reported Gbps for sanity.
- If `-p 2` had worked, the saved overhead is exactly the CPU-proxy round-trip; that is the whole point of GDAKI (GPUDirect Async Kernel-Initiated).

## Overhead analysis bullet (video log)

Three things that cost time when network data goes to the GPU:
1. **Copy path** — CPU-initiated: NIC→DRAM→(cudaMemcpy)→GPU = 2 PCIe traversals + a CPU-driven copy. GPUDirect: NIC→GPU = 1 traversal, but on this box it crosses the host-bridge interconnect (NODE).
2. **Control path** — who tells the NIC. CPU: one MMIO write. GPU `-p 1`: GPU→CPU signal + CPU MMIO write (worse than pure CPU). GPU `-p 2`: GPU MMIO write directly (best; not possible here).
3. **Synchronization** — kernel launch / host-device sync is amortized only when the kernel stays resident and keeps posting; one-shot small transfers can never win.

Numbers to quote: your latency at smallest size, bandwidth at 64 KB, and the `nvidia-smi topo -m` string.
