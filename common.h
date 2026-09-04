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

#define RTO_MS 800

#define TAIL_TIMEOUT_MS 500
#define TAIL_MAX_RETRIES 120


#define NACK_THROTTLE_MS 200


#define NACK_MIN_AGE_MS 300


#define NACK_SWEEP_CAP 512


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
