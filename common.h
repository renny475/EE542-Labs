#include <stdint.h>
#include <stddef.h>

/* Buffer capacity: big enough to carry a full jumbo-frame (MTU 9001)
 * chunk without IP fragmentation (9001 - 20 IP - 8 UDP - 16 our header
 * = 8957, leave a little headroom). The actual per-transfer chunk size
 * is a runtime parameter (see DEFAULT_PAYLOAD) so the same binaries
 * work for both the MTU 1500 and MTU 9001 test runs. */
#define MAX_PAYLOAD 8900

/* Default chunk size used when the CLI doesn't override it (sized for
 * a standard MTU 1500 path). */
#define DEFAULT_PAYLOAD 1024

/* packet_header_t.flags values */
#define FLAG_DATA 0x0
#define FLAG_EOF  0x1
#define FLAG_NACK 0x2
#define FLAG_ACK  0x3

/* Selective Repeat sliding window size (packets), shared by sender and
 * receiver. Must cover the bandwidth-delay product of the worst lab
 * case at the smallest chunk size we test (MTU 1500 -> 1024B chunks):
 * Case 2 is 100Mbit/s * 200ms RTT =~ 2.5MB, Case 3 is 80Mbit/s * 200ms
 * =~ 2MB. At 1024 bytes/packet, 1024 packets is only ~1MiB - too
 * small, and caps throughput at window_bytes/RTT regardless of actual
 * link capacity (~42Mbit/s with the old value, well under what Case
 * 2/3 can otherwise sustain). 8192 packets gives ~8MiB at the smallest
 * chunk size, comfortably over the ~2.5MB requirement with headroom
 * for Case 2's 20% loss shrinking the effective in-flight window. */
#define WINDOW_SIZE 8192

/* Sender-side per-packet retransmit timeout (fallback for when a NACK
 * doesn't catch the loss first): comfortably above the largest RTT we
 * test (200ms) so it doesn't fire spuriously ahead of a normal ACK. */
#define RTO_MS 800

/* After EOF, how long/how many times to keep polling for outstanding
 * ACKs/NACKs before giving up. Also used as the receiver's main-loop
 * poll timeout throughout (not just at the tail). */
#define TAIL_TIMEOUT_MS 500
#define TAIL_MAX_RETRIES 120

/* Minimum interval between two gap-repair sweeps (see
 * nack_window_gaps in server.c) - global, not per-seq, since a sweep
 * covers every gap currently in the window at once. */
#define NACK_THROTTLE_MS 200

/* Minimum time a gap must stay outstanding before nack_window_gaps
 * will NACK it, so a packet that's merely still in flight (sender
 * already sent it, but it hasn't arrived/been processed yet) isn't
 * mistaken for a loss. Set above the largest RTT we test (200ms) but
 * well below RTO_MS, so real losses are still caught much faster than
 * waiting for the sender's own RTO. */
#define NACK_MIN_AGE_MS 300

/* Max NACKs a single nack_window_gaps sweep will send. The lab's tc
 * qdisc is `tbf rate 100mbit latency 0.001ms burst 9015` on every hop
 * (client, server, and both router interfaces): burst 9015 bytes is
 * only ~8-9 packets, and the near-zero latency means anything past
 * that burst is dropped outright rather than queued. A sweep that
 * fires many NACKs at once (a large window can have thousands of
 * simultaneous gaps) would itself blow through that budget and
 * self-inflict more loss than it's trying to repair. */
#define NACK_SWEEP_CAP 512

/* Sender-side pacing target for data sends (Mbit/s), applied to every
 * data send - first transmission and every retransmit alike - via
 * pace_send() in client.c. Without this, the "fill window" loop fires
 * WINDOW_SIZE packets essentially instantaneously, which vastly
 * exceeds the tc tbf burst allowance above and gets almost all of it
 * dropped at the shaper instead of reaching the receiver at all. Set
 * below Case 3's 80Mbit/s router-leg cap (not just the 100Mbit/s
 * client/server egress cap) so pacing itself doesn't trigger the same
 * self-inflicted loss on that hop; leaves some margin for ACK/NACK
 * traffic sharing the link and for real network jitter. */
#define PACE_TARGET_MBPS 75

/* UDP + IPv4 header bytes added on the wire beyond our own
 * packet_header_t + payload, used by pace_send() to pace against the
 * actual bytes-on-the-wire rather than just the application payload. */
#define PACE_OVERHEAD_BYTES 28

typedef struct
{
    /* data */
    uint32_t seq_num;
    uint32_t ack_num;
    uint16_t flags;
    uint16_t data_len;
    uint32_t checksum_;
} packet_header_t;

typedef struct {

    packet_header_t header;
    char data[MAX_PAYLOAD];

} packet_t;

/* Shared checksum helpers, implemented in protocol.c. Checksum covers
 * the header (with checksum_ treated as 0) plus data_len bytes of
 * payload, so corruption of either header or data is detected. */
uint16_t compute_checksum(const void *data, size_t len);
void packet_set_checksum(packet_t *pkt);
int packet_verify_checksum(const packet_t *pkt, size_t received_len);
