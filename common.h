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
 * receiver. Sized to comfortably cover bandwidth-delay product across
 * all three lab cases (worst case ~100Mbit * 200ms RTT), so the window
 * doesn't stall waiting for ACKs before it needs to. */
#define WINDOW_SIZE 1024

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
