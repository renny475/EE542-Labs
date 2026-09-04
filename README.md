# EE542 Lab 2 - Fast, Reliable File Transfer over UDP

A custom, reliable file-transfer utility built on raw UDP sockets, designed to
compete with (and outperform) TCP under lossy, high-latency network
conditions.

## Overview

- **Transport**: UDP (IP-based, routable)
- **Reliability**: custom packet framing with sequence numbers, checksums,
  and NACK-based retransmission
- **Roles**:
  - `client.c` - the **sender**: reads a local file and streams it to the
    receiver
  - `server.c` - the **receiver**: listens for the stream and saves it to
    the output path given on the command line
- **CLI style**: modeled after `scp`

## Files

| File | Purpose |
|---|---|
| `common.h` | Shared packet format, protocol flags, and tunable constants |
| `protocol.c` | Checksum computation/verification (shared by sender and receiver) |
| `client.c` | Sender |
| `server.c` | Receiver |

## Protocol design

Each UDP datagram carries a 16-byte header (`packet_header_t` in
`common.h`) followed by up to `MAX_PAYLOAD` bytes of file data:

```c
typedef struct {
    uint32_t seq_num;    // packet sequence number
    uint32_t ack_num;    // unused
    uint16_t flags;      // FLAG_DATA / FLAG_EOF / FLAG_NACK
    uint16_t data_len;   // valid payload bytes in this packet
    uint32_t checksum_;  // covers header + payload
} packet_header_t;
```

### Sender (`client.c`)

- Splits the file into fixed-size chunks (`chunk_size`, default 1024 bytes,
  configurable via CLI for MTU 1500 vs. MTU 9001 testing).
- Sends chunks sequentially by `seq_num`. Retransmission re-reads the
  needed chunk from disk via `pread()` at `seq_num * chunk_size` - no
  in-memory buffering of already-sent packets is needed, since the file
  itself (indexed by sequence number) acts as the retransmit source.
- After the last chunk, sends an `EOF` packet carrying the total packet
  count, then enters a bounded tail-wait phase (retrying `EOF` and
  servicing any further NACKs) until the receiver goes quiet.

### Receiver (`server.c`)

- Tracks which sequence numbers have arrived in a bitmap (`received[]`)
  and writes each packet's payload directly to the output file at the
  correct offset via `pwrite()` - no reordering buffer is needed since
  the offset is derived from the sequence number.
- Maintains `expected_seq`, the lowest sequence number not yet received.
  When a later packet arrives while `expected_seq` is still missing (real
  evidence of loss or reordering, not just "hasn't arrived yet"), it
  triggers a repair sweep.
- **Batched gap repair**: rather than requesting one missing packet at a
  time (which cannot keep up once loss produces thousands of simultaneous
  gaps), the receiver periodically (every `NACK_THROTTLE_MS`) sweeps
  forward from `expected_seq` and NACKs up to `NACK_SWEEP_CAP` outstanding
  gaps in a single pass. This is what lets the protocol recover under
  Case 2's 20% loss / 200ms RTT within a few minutes instead of hours.
- After `EOF` is received, the same sweep continues on a timeout-driven
  cadence (bounded by `TAIL_MAX_RETRIES`) until every gap is closed or the
  receiver gives up.
- Verifies a checksum on every packet (covering header + payload), so
  corruption is treated the same as loss.

## Building

```bash
gcc -Wall -Wextra -o client client.c protocol.c
gcc -Wall -Wextra -o server server.c protocol.c
```

## Usage

Start the receiver first:

```bash
./server 9999 received_case3.bin
```

Then run the sender:

```bash
./client 192.168.10.100 9999 testfile.bin
```

### Verifying a transfer

```bash
md5sum <file_to_send>
md5sum <output_file>
```

Matching hashes confirm the file was delivered without errors, even under
packet loss.

## Test environment

Three VMs connected as **client -- VyOS router -- server**, with:

- `client`/`server`: egress rate-limited to 100Mbit/s via `tc tbf`
- `router`: rate limiting + `netem` delay/loss applied per test case on
  both data-path interfaces

## Results (1GB file, MTU 1500)

## Results (1GB file, MTU 9100)


## Pending

- [ ] Re-run Case 1/2/3 at MTU 9001 with a larger `chunk_size`
- [ ] Bidirectional iperf TCP/UDP baseline data for Case 2 and Case 3 (for
      the Critical Thinking report section)
- [ ] Selective Repeat sliding window with explicit sender/receiver
      memory buffers (in progress - a prior attempt had a stall bug under
      loss and was reverted; the current version above is the last
      validated one)
