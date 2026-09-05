#include <stdint.h>
#include <stddef.h>

#define MAX_PAYLOAD 8900


#define DEFAULT_PAYLOAD 1456

/* packet_header_t.flags values */
#define FLAG_DATA 0x0
#define FLAG_EOF  0x1
#define FLAG_NACK 0x2
#define FLAG_ACK  0x3


#define WINDOW_SIZE 8192

/* Lowered from 800 to 400 (2x the largest real RTT we test, 200ms):
 * with the NACK path now behaving correctly (see NACK_SWEEP_CAP and
 * the pending_since reset in nack_window_gaps), RTO no longer needs
 * such a large margin to avoid false positives, and a shorter fallback
 * matters a lot under Case 2's 20% loss, where a meaningful fraction
 * of packets end up needing it regardless of NACK. Measured on a local
 * Case 2 emulation (100Mbit, 200ms RTT, 20% bidirectional loss): 800ms
 * -> 9.3 Mbit/s, 500ms -> 14.3 Mbit/s, 400ms -> 16.4 Mbit/s, with
 * Case 1 and Case 3 also improving (not regressing) at 400ms. */
#define RTO_MS 400

#define TAIL_TIMEOUT_MS 500
#define TAIL_MAX_RETRIES 120


#define NACK_THROTTLE_MS 200


#define NACK_MIN_AGE_MS 300


/* Raised to WINDOW_SIZE: a sweep can now safely address every gap in
 * the window in one pass because nack_window_gaps paces its sends
 * (see pace_nack_send() in server.c) instead of firing them in a tight
 * loop. The old cap of 512 was itself unsafe: 512 tiny (~44 byte)
 * NACKs sent back to back is ~22.5KB, well over the tc tbf burst
 * allowance (~9015 bytes, good for only ~204 NACK-sized packets
 * instantaneously) - so under Case 2's 20% loss, a dense sweep was
 * silently losing NACKs to the same burst-drop mechanism pacing exists
 * to prevent elsewhere, on top of only covering under a third of a
 * fully-loaded window's gaps per pass to begin with. */
#define NACK_SWEEP_CAP WINDOW_SIZE

/* Pacing for NACK sends within a sweep (see pace_nack_send() in
 * server.c), same rationale and target as the sender's data pacing:
 * batch sends and cap each batch under the tc tbf burst allowance so a
 * sweep with many gaps can't blow through it the way an uncapped tight
 * loop would. NACK_PACE_BATCH_BYTES is sized in NACK-sized units (~44
 * bytes each) rather than reusing PACE_BATCH_BYTES, which was sized
 * for full data packets. */
#define NACK_PACE_TARGET_MBPS 75
#define NACK_PACE_BATCH_BYTES 8000


#define PACE_TARGET_MBPS 75


#define PACE_OVERHEAD_BYTES 28


#define PACE_BATCH_BYTES 8000

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
