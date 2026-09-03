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

/* Tail-wait phase: after EOF, how long/how many times to keep polling
 * for outstanding NACKs/retransmits before giving up. */
#define TAIL_TIMEOUT_MS 500
#define TAIL_MAX_RETRIES 120

/* Minimum interval between two gap-repair sweeps (see periodic_nack_sweep
 * in server.c) - not per-seq, global, since a sweep now covers many gaps
 * at once instead of chasing a single lowest gap. */
#define NACK_THROTTLE_MS 200

/* Max number of NACKs sent in a single sweep. Kept well under the tbf
 * burst allowance (9015 bytes / 16-byte NACKs) so a sweep can't flood
 * past the router's token bucket and self-inflict extra loss. */
#define NACK_SWEEP_CAP 512

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
