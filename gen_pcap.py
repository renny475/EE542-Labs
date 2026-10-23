import argparse, random, struct
import numpy as np
from scapy.all import Ether, IP, UDP, Raw, wrpcap

MAGIC = 0x45453534
HDR = "!IIHHIHI"
HDR_LEN = struct.calcsize(HDR)
CHUNK = 1400
H = W = 224; C = 3
FRAME_BYTES = H * W * C

def make_images(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, H, W, C), dtype=np.uint8)

def packetize(frame_id, img):
    data = img.tobytes()
    nseg = (len(data) + CHUNK - 1) // CHUNK
    pkts = []
    for seq in range(nseg):
        off = seq * CHUNK
        chunk = data[off:off + CHUNK]
        hdr = struct.pack(HDR, MAGIC, frame_id, seq, nseg, off, len(chunk), len(data))
        pkt = (Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") /
               IP(src="10.0.0.1", dst="10.0.0.2") /
               UDP(sport=40000, dport=5000) / Raw(hdr + chunk))
        pkts.append(pkt)
    return pkts

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="images.pcap")
    a = ap.parse_args()

    imgs = make_images(a.n, a.seed)
    np.save("images.npy", imgs)

    packets = [p for fid in range(a.n) for p in packetize(fid, imgs[fid])]
    if a.shuffle:
        random.Random(a.seed).shuffle(packets)
    wrpcap(a.out, packets)

    assert HDR_LEN == 22
    assert all(len(bytes(p)) <= 1514 for p in packets)
    print(f"{a.n} images -> {len(packets)} packets ({len(packets)//a.n}/image), "
          f"max frame {max(len(bytes(p)) for p in packets)} B, shuffled={a.shuffle}, wrote {a.out}")
